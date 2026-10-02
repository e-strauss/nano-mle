import type { Metadata } from "next";
import LoginForm from "./login-form";

export const metadata: Metadata = { title: "Sign in · ML search dashboard" };

export default function LoginPage() {
  return (
    <main className="login">
      <h1>ML search dashboard</h1>
      <LoginForm />
    </main>
  );
}
