import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SourceIQ | Enterprise Knowledge Assistant",
  description:
    "Ask questions across your company documents and get grounded answers with page-level citations.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
