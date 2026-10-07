import { LINKS } from "../lib/links";

/** Shell footer: project links + operating-point note. */
export function Footer() {
  const item =
    "font-mono text-xs tracking-wider text-dim transition-colors hover:text-teal";
  return (
    <footer className="border-t border-hairline">
      <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-x-6 gap-y-2 px-4 py-5 md:px-6">
        <span className="font-mono text-xs tracking-wider text-dim">
          SIGNAL LAB // FORENSIC VOICE CONSOLE
        </span>
        <span className="ml-auto flex flex-wrap gap-x-5 gap-y-2">
          <a className={item} href={LINKS.github} target="_blank" rel="noreferrer">
            GITHUB
          </a>
          <a className={item} href={LINKS.spaceApi} target="_blank" rel="noreferrer">
            SPACE API
          </a>
          <a
            className={item}
            href={LINKS.modelWeights}
            target="_blank"
            rel="noreferrer"
          >
            MODEL WEIGHTS
          </a>
        </span>
      </div>
    </footer>
  );
}
