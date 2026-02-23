import { NextAuthOptions } from "next-auth";
import CredentialsProvider from "next-auth/providers/credentials";
import GoogleProvider from "next-auth/providers/google";
import FacebookProvider from "next-auth/providers/facebook";
import { PrismaAdapter } from "@auth/prisma-adapter";
import bcrypt from "bcryptjs";
import { prisma } from "./prisma";

export const authOptions: NextAuthOptions = {
  adapter: PrismaAdapter(prisma) as NextAuthOptions["adapter"],
  providers: [
    ...(process.env.GOOGLE_CLIENT_ID && process.env.GOOGLE_CLIENT_SECRET
      ? [GoogleProvider({ clientId: process.env.GOOGLE_CLIENT_ID, clientSecret: process.env.GOOGLE_CLIENT_SECRET })]
      : []),
    ...(process.env.FACEBOOK_CLIENT_ID && process.env.FACEBOOK_CLIENT_SECRET
      ? [FacebookProvider({ clientId: process.env.FACEBOOK_CLIENT_ID, clientSecret: process.env.FACEBOOK_CLIENT_SECRET })]
      : []),
    CredentialsProvider({
      name: "credentials",
      credentials: {
        email: { label: "אימייל", type: "email" },
        password: { label: "סיסמה", type: "password" },
      },
      async authorize(credentials) {
        if (!credentials?.email || !credentials?.password) return null;

        const user = await prisma.user.findUnique({
          where: { email: credentials.email },
        });

        if (!user || !user.password) return null;

        const valid = await bcrypt.compare(credentials.password, user.password);
        if (!valid) return null;

        return {
          id: user.id,
          name: user.name,
          email: user.email,
          role: user.role,
        };
      },
    }),
  ],
  callbacks: {
    async jwt({ token, user }) {
      if (user) {
        // On first sign in, load role from DB
        const dbUser = await prisma.user.findUnique({
          where: { email: user.email! },
        });
        token.role = dbUser?.role ?? "STUDENT";
        token.id = user.id;
      }
      return token;
    },
    async session({ session, token }) {
      if (session.user) {
        (session.user as { id: string; role: string }).id = token.sub!;
        (session.user as { role: string }).role = token.role as string;
      }
      return session;
    },
  },
  events: {
    async createUser({ user }) {
      // New OAuth user - assign STUDENT role by default
      await prisma.user.update({
        where: { id: user.id },
        data: { role: "STUDENT" },
      });
    },
  },
  pages: {
    signIn: "/login",
  },
  session: { strategy: "jwt" },
  secret: process.env.NEXTAUTH_SECRET,
};
