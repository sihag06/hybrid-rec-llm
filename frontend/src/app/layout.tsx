import "./globals.css";

export const metadata = {
  title: "SHL Assessment Recommender",
  description: "Chat + recommendations UI powered by FastAPI backend",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="bg-slate-100">{children}</body>
    </html>
  )
}
