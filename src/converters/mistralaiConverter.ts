import { App, Notice, TFile } from 'obsidian';
import { Mistral } from '@mistralai/mistralai';
import { MarkerSettings } from '../settings';
import { BaseConverter, ConversionResult } from '../converter';
import { ConverterSettingDefinition } from '../utils/converterSettingsUtils';
import { deleteOriginalFile, checkForExistingFiles } from '../utils/fileUtils';
import { OCRPageObject } from '@mistralai/mistralai/models/components';

const CAVEMAN_SYSTEM_PROMPT_BASE = `You are a text annotation and reformatting assistant.
You will receive a markdown document produced by OCR. Your task is two-fold:
1. Rewrite the text portions using the caveman communication style described below (leave code blocks, URLs, file names, and technical identifiers unchanged).
2. Annotate key concepts inline using the label system described below.

Do NOT add extra headings, preamble, or explanation. Return only the reformatted markdown.

## Caveman Style Rules

Drop: articles (a/an/the), filler words (just/really/basically/actually/simply), pleasantries (sure/certainly/of course/happy to), hedging phrases. Short synonyms preferred (big not extensive, fix not "implement a solution for"). Use arrows (→) for causality (X → Y). Technical terms, code symbols, function names, and API names stay exact.

Pattern: [thing] [action] [reason]. [next step].

## Intensity Levels`;

const CAVEMAN_LEVEL_PROMPTS: Record<string, string> = {
  lite: `${CAVEMAN_SYSTEM_PROMPT_BASE}

Current level: **lite** — No filler or hedging. Keep articles and full sentences. Professional but tight.`,

  full: `${CAVEMAN_SYSTEM_PROMPT_BASE}

Current level: **full** — Drop articles, fragments OK, short synonyms, arrows for causality (→). Classic caveman.`,

  ultra: `${CAVEMAN_SYSTEM_PROMPT_BASE}

Current level: **ultra** — Abbreviate prose words (DB/auth/config/req/res/fn/impl), strip conjunctions, arrows for causality (→), one word when one word enough. Code symbols, function names, API names, error strings: never abbreviate.`,
};

const ANNOTATION_LABELS_PROMPT = `
## Annotation Label System

Prepend labels to sentences/paragraphs that match the trigger keywords. Format depends on intensity level:
- lite:  [emoji]**[Word]:** (full word, bold) — e.g. 📣**Claim:** or ✅**Answer:**
- full:  [emoji]**[Abbr]:** (abbreviated, bold) — e.g. 📣**Clm:** or ✅**Ans:**
- ultra: [emoji][Abbr]: (no bold) — e.g. 📣Clm: or ✅Ans:

The opening ** must appear immediately after the emoji and the closing ** must appear immediately after the colon, like this: [emoji]**[Word/Abbr]:**

Labels, abbreviations, and trigger keywords:
❓Q(uestion): question/ask/unclear
🔓OpenQ: open question/unsolved/future work
✅Ans(wer): answered/resolved
📣Claim: claim/assert/authors say
🔭Hyp(othesis): hypothesis/predict/conjecture
💭Assume: assume/unverified/paper assumes/premise
📊Result: result/finding/outcome/showed
🔩Mech(anism): mechanism/how it works/why/underlying
📐Meth(od): methodology/method/approach/procedure
🔁Analogy: analogy/maps to/equivalent/like
🧪Test: test/experiment/ablation/validate
👍Pro: pro/benefit/advantage/upside
👎Con: con/downside/drawback/cost
🧱Lim(itation): limitation/constraint/caveat/cannot
⚠️Warn(ing): warning/danger/risky/beware
🚫Not: wrong/incorrect/false/misconception
⚡Contr(adiction): contradicts/conflicts/inconsistent
💡Idea: idea/suggest/propose/direction
🔧Fix: fix/patch/debug/repair
❗Imp(ortant): important/critical/must/crucial
🔍Check: verify/unsure/confirm/look up
🗝️Key: key insight/takeaway/core/essential
📚Ref: reference/cite/paper/source
ℹ️Info: info/context/background/fyi/general
💬Talk: quote/said/mentioned/according to/discussion
🟢Ok: works/valid/confirmed/acceptable
🔴NotOk: broken/fails/invalid/rejected
✍️Write: draft/document/todo-write
⭐Star: notable/remarkable/highlight/standout
🤖AI: ai-generated/model/llm/gpt/claude
🥇Best/🥈2nd/🥉3rd: rankings/top/winner
🔗Link: url/connect/related to/see also
⏳Time: duration/deadline/epoch/when
⚙️Set: config/hyperparameter/param/setting
✔️Done: complete/finished/closed/resolved
🎯Goal: goal/objective/aim/purpose/target
🗄️Data: data/dataset/corpus/benchmark/annotation

Only add labels where content clearly matches a trigger. Do not force-label every sentence.
`;

// All known annotation label words and abbreviations, sorted longest-first so
// regex alternation is greedy (e.g. "OpenQ" matches before "Q").
const ANNOTATION_LABEL_WORDS = [
  'Question', 'OpenQ', 'Answer', 'Claim', 'Hypothesis', 'Assume', 'Result',
  'Mechanism', 'Method', 'Analogy', 'Test', 'Pro', 'Con', 'Limitation',
  'Warning', 'Not', 'Contradiction', 'Idea', 'Fix', 'Important', 'Check',
  'Key', 'Ref', 'Info', 'Talk', 'Ok', 'NotOk', 'Write', 'Star', 'AI',
  'Best', '2nd', '3rd', 'Link', 'Time', 'Set', 'Done', 'Goal', 'Data',
  // abbreviations used by full / ultra
  'Q', 'Ans', 'Clm', 'Hyp', 'Mech', 'Meth', 'Lim', 'Warn', 'Contr', 'Imp',
].sort((a, b) => b.length - a.length);

// Pre-compiled regex for annotation label normalisation.
// Matches: EMOJI + optional variation selector + optional whitespace +
//          optional ** + LABEL + : + optional **
// e.g. "📣**Clm:**", "📣Clm:", "⚠️ **Warn:**"
const ANNOTATION_LABEL_PATTERN = new RegExp(
  `([^\\u0000-\\u007F]\\uFE0F?)\\s*\\*{0,2}(${ANNOTATION_LABEL_WORDS.join('|')}):\\*{0,2}`,
  'gu'
);

export class MistralAIConverter extends BaseConverter {
  async convert(
    app: App,
    settings: MarkerSettings,
    file: TFile
  ): Promise<boolean> {
    const folderPath = await this.prepareConversion(settings, file);
    if (!folderPath) return false;

    if (
      (settings.extractContent === 'images' ||
        settings.extractContent === 'all') &&
      !(await checkForExistingFiles(app, folderPath))
    ) {
      return true;
    }

    if (!settings.mistralaiApiKey) {
      new Notice('Error: MistralAI API key is not configured');
      console.error('Missing MistralAI API key in settings');
      return false;
    }

    new Notice('Converting file with MistralAI OCR...', 4000);

    const client = new Mistral({ apiKey: settings.mistralaiApiKey });
    let uploadedFileId: string | undefined;

    try {
      // Read the file content
      const fileContent = await app.vault.readBinary(file);

      // Upload the file to MistralAI
      new Notice('Uploading file to MistralAI...', 2000);
      const fileUpload = await client.files.upload({
        file: {
          fileName: file.name,
          content: fileContent,
        },
        purpose: 'ocr',
      });

      if (!fileUpload || !fileUpload.id) {
        new Notice('Failed to upload file to MistralAI');
        return false;
      }

      uploadedFileId = fileUpload.id;

      const signedUrl = await client.files.getSignedUrl({
        fileId: uploadedFileId,
      });

      // Set includeImageBase64 based on the extractContent setting
      const includeImages = settings.extractContent !== 'text';

      const imageLimit =
        (settings.imageLimit ?? 0) > 0 ? settings.imageLimit : undefined;

      // Add image min size if set
      const imageMinSize =
        (settings.imageMinSize ?? 0) > 0 ? settings.imageMinSize : undefined;

      const ocrResponse = await client.ocr.process({
        model: 'mistral-ocr-latest',
        document: {
          type: 'document_url',
          documentUrl: signedUrl.url,
        },
        includeImageBase64: includeImages,
        imageLimit: imageLimit,
        imageMinSize: imageMinSize,
      });

      if (!ocrResponse || !ocrResponse.pages) {
        new Notice('Failed to process file with OCR');
        return false;
      }

      // Parse OCR results
      let conversionResult = this.parseOCRResults(
        ocrResponse.pages,
        settings.extractContent
      );

      // Post-process with caveman annotation style if configured
      const annotationStyle = settings.mistralaiAnnotationStyle || 'none';
      if (
        conversionResult.success &&
        conversionResult.markdown &&
        annotationStyle !== 'none'
      ) {
        new Notice(
          `Applying annotation style (${annotationStyle}) with MistralAI...`,
          3000
        );
        conversionResult = await this.postProcessWithCavemanStyle(
          client,
          conversionResult,
          annotationStyle
        );
      }

      // Process the conversion result
      await this.processConversionResult(
        app,
        settings,
        conversionResult,
        folderPath,
        file
      );

      new Notice('MistralAI OCR conversion completed successfully');

      if (settings.deleteOriginal) {
        await deleteOriginalFile(app, file);
      }

      return true;
    } catch (error) {
      console.error('MistralAI conversion error:', error.message, error.stack);
      new Notice(
        `MistralAI conversion failed: ${
          error.message || 'Network or server error'
        }`
      );
      return false;
    } finally {
      if (
        settings.deleteFileFromMistralaiAfterConversion &&
        uploadedFileId
      ) {
        try {
          const deleteResult = await client.files.delete({
            fileId: uploadedFileId,
          });

          if (!deleteResult?.deleted) {
            console.warn(
              `MistralAI file deletion returned non-deleted status for file ${uploadedFileId}`,
              deleteResult
            );
            new Notice(
              'Warning: Uploaded MistralAI file may not have been deleted.'
            );
          }
        } catch (cleanupError) {
          console.error(
            `Failed to delete uploaded MistralAI file ${uploadedFileId}:`,
            cleanupError
          );
          new Notice(
            'Warning: Failed to delete uploaded file from MistralAI after conversion.'
          );
        }
      }
    }
  }

  private async postProcessWithCavemanStyle(
    client: Mistral,
    conversionResult: ConversionResult,
    level: string
  ): Promise<ConversionResult> {
    try {
      const systemPrompt =
        (CAVEMAN_LEVEL_PROMPTS[level] || CAVEMAN_LEVEL_PROMPTS['lite']) +
        ANNOTATION_LABELS_PROMPT +
        `\n\nIMPORTANT: The text may contain image placeholder tokens of the form __IMG_0__, __IMG_1__, etc. These are sentinels for embedded images. You MUST reproduce every such token exactly as-is, in its original position, without modification.`;

      const markdown = conversionResult.markdown || '';

      // --- Symptom 1: extract image tags → sentinels ---
      const imageTokens: string[] = [];
      const sanitized = markdown.replace(
        /!\[[^\]]*\]\([^)]*\)/g,
        (match) => {
          const idx = imageTokens.length;
          imageTokens.push(match);
          return `__IMG_${idx}__`;
        }
      );

      // --- Symptom 2: batch pages to stay under ~6 000-token limit ---
      const PAGE_SEP = '\n\n---\n\n';
      const TOKEN_BUDGET = 6000;
      // 4 chars/token is a rough approximation; adjust if content is code-heavy
      const CHARS_PER_TOKEN = 4;
      const CHAR_BUDGET = TOKEN_BUDGET * CHARS_PER_TOKEN;

      const pages = sanitized.split(PAGE_SEP);
      const batches: string[][] = [];
      let currentBatch: string[] = [];
      let currentLen = 0;

      for (const page of pages) {
        if (
          currentBatch.length > 0 &&
          currentLen + page.length > CHAR_BUDGET
        ) {
          batches.push(currentBatch);
          currentBatch = [];
          currentLen = 0;
        }
        currentBatch.push(page);
        currentLen += page.length;
      }
      if (currentBatch.length > 0) {
        batches.push(currentBatch);
      }

      const processedBatches: string[] = [];
      for (const batch of batches) {
        const batchText = batch.join(PAGE_SEP);
        const response = await client.chat.complete({
          model: 'mistral-small-latest',
          messages: [
            { role: 'system', content: systemPrompt },
            { role: 'user', content: batchText },
          ],
        });

        const result = response?.choices?.[0]?.message?.content;
        if (typeof result !== 'string' || !result.trim()) {
          console.warn(
            'MistralAI annotation: empty/unexpected response for batch, keeping original text'
          );
        }
        processedBatches.push(
          typeof result === 'string' && result.trim() ? result : batchText
        );
      }

      // Reassemble batches with page separator, then normalize bold formatting.
      // Each batch is an independent LLM call, so the model may apply bold
      // inconsistently across batches. The regex below is the single source of
      // truth: it enforces the correct bold style for the selected level on
      // every annotation label, regardless of what the LLM decided to do.
      let processed = this.normalizeAnnotationBold(
        processedBatches.join(PAGE_SEP),
        level
      );

      // --- Restore image sentinels → original tags ---
      processed = processed.replace(/__IMG_(\d+)__/g, (_, idx) => {
        const original = imageTokens[parseInt(idx, 10)];
        return original !== undefined ? original : `__IMG_${idx}__`;
      });

      return {
        ...conversionResult,
        markdown: processed,
      };
    } catch (error) {
      console.error('MistralAI annotation post-processing error:', error);
      new Notice(
        `Annotation post-processing failed: ${error.message || 'Unknown error'}. Using original OCR output.`
      );
    }
    return conversionResult;
  }

  /**
   * Deterministically normalize annotation label bold formatting.
   *
   * Each batch is processed by an independent LLM call, so the model may
   * apply (or omit) `**bold**` inconsistently across batches.  This method
   * is the single source of truth: it scans for every known annotation label
   * following an emoji and enforces the style required by `level`:
   *   lite / full  →  emoji**Label:**
   *   ultra        →  emojiLabel:
   */
  private normalizeAnnotationBold(text: string, level: string): string {
    // Use the pre-compiled pattern; reset lastIndex before each use because
    // the global flag keeps state between calls.
    ANNOTATION_LABEL_PATTERN.lastIndex = 0;
    if (level === 'ultra') {
      // ultra: no bold
      return text.replace(ANNOTATION_LABEL_PATTERN, '$1$2:');
    }
    // lite / full: enforce bold immediately after emoji, no space
    ANNOTATION_LABEL_PATTERN.lastIndex = 0;
    return text.replace(ANNOTATION_LABEL_PATTERN, '$1**$2:**');
  }

  private parseOCRResults(
    pages: OCRPageObject[],
    extractContent = 'all'
  ): ConversionResult {
    try {
      // Combine all pages into a single markdown string
      let markdown = '';
      const images: { [key: string]: string } = {};

      // Process each page
      pages.forEach((page, index) => {
        // Add page separator if paginate is enabled (we'll check in processConversionResult)
        if (index > 0) {
          markdown += '\n\n---\n\n';
        }

        // Only include text content if extractContent isn't set to 'images'
        if (extractContent !== 'images') {
          // Add page content
          markdown += page.markdown || '';
        }

        // Only process images if extractContent isn't set to 'text'
        if (
          extractContent !== 'text' &&
          page.images &&
          page.images.length > 0
        ) {
          page.images.forEach((image) => {
            // Create unique image name with page number prefix
            const imageName = image.id;

            // Strip the data URL prefix if it exists
            let base64Data = image.imageBase64 || '';
            if (base64Data.startsWith('data:')) {
              // Remove the prefix (e.g., 'data:image/jpeg;base64,')
              base64Data = base64Data.split(',')[1];
            }

            images[imageName] = base64Data;
          });
        }
      });

      return {
        success: true,
        markdown,
        images,
        metadata: {
          page_count: pages.length,
          processor: 'mistralai-ocr',
        },
      };
    } catch (error) {
      console.error('Error parsing OCR results:', error);
      return {
        success: false,
        error: `Failed to parse OCR results: ${error.message}`,
      };
    }
  }

  async testConnection(
    settings: MarkerSettings,
    silent: boolean | undefined
  ): Promise<boolean> {
    if (!settings.mistralaiApiKey) {
      if (!silent) new Notice('Error: MistralAI API key is not configured');
      return false;
    }

    try {
      // Initialize MistralAI client
      const client = new Mistral({ apiKey: settings.mistralaiApiKey });

      // Make a simple API call to test the connection
      // We'll just list the models to see if the API key is valid and the connection is successful
      const response = await client.files.list();

      if (response) {
        if (!silent) new Notice('MistralAI connection successful!');
        return true;
      }

      if (!silent) new Notice('Error connecting to MistralAI API');
      return false;
    } catch (error) {
      if (!silent) {
        new Notice(`Error connecting to MistralAI API: ${error.message}`);
      }
      console.error('Error connecting to MistralAI API:', error);
      return false;
    }
  }

  getConverterSettings(): ConverterSettingDefinition[] {
    return [
      {
        id: 'mistralaiApiKey',
        name: 'MistralAI API Key',
        description: 'Enter your MistralAI API key',
        type: 'text',
        placeholder: 'API Key',
        defaultValue: '',
        buttonText: 'Test connection',
        buttonAction: async (app, settings) => {
          await this.testConnection(settings, false);
        },
      },
      {
        id: 'deleteFileFromMistralaiAfterConversion',
        name: 'Delete file from mistralai after conversion',
        description:
          'Delete uploaded files from the MistralAI API after each conversion.',
        type: 'toggle',
        defaultValue: false,
      },
      {
        id: 'imageLimit',
        name: 'Image limit',
        description: 'Maximum number of images to extract (0 for no limit)',
        type: 'text',
        placeholder: '0',
        defaultValue: '0',
        onChange: async (value, settings) => {
          const numValue = value ? parseInt(value) : 0;
          settings.imageLimit = isNaN(numValue) ? 0 : numValue;
        },
      },
      {
        id: 'imageMinSize',
        name: 'Image minimum size',
        description:
          'Minimum height and width of images to extract (0 for no minimum)',
        type: 'text',
        placeholder: '0',
        defaultValue: '0',
        onChange: async (value, settings) => {
          const numValue = value ? parseInt(value) : 0;
          settings.imageMinSize = isNaN(numValue) ? 0 : numValue;
        },
      },
      {
        id: 'paginate',
        name: 'Paginate',
        description: 'Add horizontal rules between each page',
        type: 'toggle',
        defaultValue: false,
      },
      {
        id: 'mistralaiAnnotationStyle',
        name: 'Annotation style',
        description:
          'Post-process extracted text with caveman compression and annotation labels using the MistralAI chat API. "None" keeps the original OCR output.',
        type: 'dropdown',
        defaultValue: 'none',
        options: [
          { value: 'none', label: 'None (keep original)' },
          { value: 'lite', label: 'Lite – no filler, full sentences' },
          { value: 'full', label: 'Full – fragments, arrows, short synonyms' },
          { value: 'ultra', label: 'Ultra – maximum compression' },
        ],
      },
    ];
  }
}
