"use client";

import { useRouter } from "next/navigation";
import { useEffect } from "react";

// Re-render the server page periodically while something is live.
export default function AutoRefresh({ active, seconds = 5 }: { active: boolean; seconds?: number }) {
  const router = useRouter();
  useEffect(() => {
    if (!active) return;
    const t = setInterval(() => router.refresh(), seconds * 1000);
    return () => clearInterval(t);
  }, [active, seconds, router]);
  return active ? <span className="badge live">live · refreshing every {seconds}s</span> : null;
}
