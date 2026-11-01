import { defineConfig } from 'vitest/config';
import { fileURLToPath } from 'node:url';

const shared = (f: string) => fileURLToPath(new URL(`./vendor/shared/${f}.ts`, import.meta.url));

export default defineConfig({
  resolve: {
    alias: {
      '@vspark/shared/signal_types': shared('signal_types'),
      '@vspark/shared/signal': shared('signal'),
      '@vspark/shared/node_decorators': shared('node_decorators'),
      '@vspark/shared/node': shared('node'),
    },
  },
  test: { include: ['test/**/*.test.ts'] },
});
