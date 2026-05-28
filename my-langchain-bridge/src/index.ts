import { type PluginContext } from "@lmstudio/sdk";

export async function main(context: PluginContext) {
  // Use the native SDK builder method to hook up your custom text generation pipeline
  context.withGenerator(async (ctl, chat) => {
    // Grab the last message the user typed into the UI
    const lastUserMessage = chat.at(-1).getText();

    try {
      // Send it to your running Python FastAPI backend
      const response = await fetch("http://127.0.0.1:8000/generate-stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: lastUserMessage }),
      });

      if (!response.body) {
        ctl.fragmentGenerated("Error: No data stream received from FastAPI backend.");
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      // Stream tokens directly into the LM Studio chat window
      while (true) {
        if (ctl.abortSignal.aborted) {
          await reader.cancel();
          break;
        }

        const { value, done } = await reader.read();
        if (done) break;

        const textChunk = decoder.decode(value, { stream: true });
        ctl.fragmentGenerated(textChunk);
      }
    } catch (error) {
      ctl.fragmentGenerated(`\n[Backend Connection Error]: ${(error as Error).message}`);
    }
  });
}