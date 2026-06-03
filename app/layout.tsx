import type { Metadata } from 'next'
import './globals.css'
export const metadata: Metadata = { title: 'RAGAS Evaluation Lab', description: 'Paper-based RAGAS evaluation dashboard.' }
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return <html lang="fa" dir="rtl"><body>{children}</body></html>
}
