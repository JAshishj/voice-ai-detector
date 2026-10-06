import { LANGUAGES, type Language } from "../lib/api";

interface Props {
  value: Language;
  onChange: (lang: Language) => void;
  disabled?: boolean;
}

const CODES: Record<Language, string> = {
  Tamil: "TA-IN",
  English: "EN-US",
  Hindi: "HI-IN",
  Malayalam: "ML-IN",
  Telugu: "TE-IN",
};

/** Acoustic profile selector — mono uppercase label above, chip group. */
export function LanguageSelect({ value, onChange, disabled }: Props) {
  return (
    <fieldset disabled={disabled}>
      <legend className="font-mono text-xs font-medium tracking-[0.08em] text-dim">
        ACOUSTIC MODEL &amp; LANGUAGE PROFILE
      </legend>
      <div className="mt-3 flex flex-wrap gap-2" role="radiogroup">
        {LANGUAGES.map((lang) => {
          const active = lang === value;
          return (
            <button
              key={lang}
              type="button"
              role="radio"
              aria-checked={active}
              onClick={() => onChange(lang)}
              className={`min-h-[44px] rounded-full border px-4 py-2 text-sm transition-colors ${
                active
                  ? "border-teal bg-teal font-semibold text-void"
                  : "border-hairline bg-panel text-ink hover:border-teal/60"
              }`}
            >
              {lang}{" "}
              <span
                className={`font-mono text-[11px] ${active ? "text-void/70" : "text-dim"}`}
              >
                {CODES[lang]}
              </span>
            </button>
          );
        })}
      </div>
    </fieldset>
  );
}
