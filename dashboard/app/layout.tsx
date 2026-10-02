import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ML search dashboard",
  description: "Runs, search trees and launches of ML experiment harnesses",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
