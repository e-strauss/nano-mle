import Link from "next/link";
import { logout } from "@/app/actions/auth";

export default function Header() {
  return (
    <header className="top">
      <Link href="/" className="brand">ML search dashboard</Link>
      <Link href="/">Runs</Link>
      <Link href="/new">New run</Link>
      <span className="spacer" />
      <form action={logout}><button type="submit">Sign out</button></form>
    </header>
  );
}
