const { PrismaClient } = require('@prisma/client');
const prisma = new PrismaClient();

async function main() {
  console.log('Seeding model metrics...');

  // Logistic Regression Metrics (from 02_ML.ipynb)
  await prisma.modelMetric.upsert({
    where: { modelName: 'Logistic Regression' },
    update: {
      accuracy: 90.15,
      precision: 90.1,
      recall: 90.2,
      f1: 90.1,
      params: 50000,
      sizeBytes: 1500000,
      inferenceMs: 2,
    },
    create: {
      modelName: 'Logistic Regression',
      accuracy: 90.15,
      precision: 90.1,
      recall: 90.2,
      f1: 90.1,
      params: 50000,
      sizeBytes: 1500000,
      inferenceMs: 2,
    },
  });

  // Bi-LSTM Metrics (from LSTM.ipynb)
  await prisma.modelMetric.upsert({
    where: { modelName: 'Bi-LSTM' },
    update: {
      accuracy: 84.67,
      precision: 85.0,
      recall: 84.5,
      f1: 84.7,
      params: 120000,
      sizeBytes: 5000000,
      inferenceMs: 22,
    },
    create: {
      modelName: 'Bi-LSTM',
      accuracy: 84.67,
      precision: 85.0,
      recall: 84.5,
      f1: 84.7,
      params: 120000,
      sizeBytes: 5000000,
      inferenceMs: 22,
    },
  });

  // BERT Metrics (from full 25k test set evaluation)
  await prisma.modelMetric.upsert({
    where: { modelName: 'BERT' },
    update: {
      accuracy: 91.68,
      precision: 93.32,
      recall: 89.79,
      f1: 91.52,
      params: 110000000,
      sizeBytes: 420000000,
      inferenceMs: 72,
    },
    create: {
      modelName: 'BERT',
      accuracy: 91.68,
      precision: 93.32,
      recall: 89.79,
      f1: 91.52,
      params: 110000000,
      sizeBytes: 420000000,
      inferenceMs: 72,
    },
  });

  console.log('Model metrics seeded successfully.');
}

main()
  .catch((e) => {
    console.error(e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
