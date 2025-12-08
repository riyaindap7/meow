import { betterAuth } from "better-auth";
import { MongoClient } from "mongodb";
import { mongodbAdapter } from "better-auth/adapters/mongodb";

const mongoUri = process.env.MONGODB_URI || "mongodb://admin:meow@192.168.0.106:27017/";
const databaseName = "victor_rag";

const client = new MongoClient(mongoUri);
const db = client.db(databaseName);

export const auth = betterAuth({
  database: mongodbAdapter(db, {
    client,
    // Disable transactions for standalone MongoDB
    transaction: false
  }),
  emailAndPassword: {
    enabled: true,
    requireEmailVerification: false,
  },
  session: {
    expiresIn: 60 * 60 * 24 * 7, // 7 days
    updateAge: 60 * 60 * 24, // 1 day
  },
  // Add secret key
  secret: process.env.BETTER_AUTH_SECRET || "your-secret-key-here-make-it-very-long-and-random",
});

export type Session = typeof auth.$Infer.Session;