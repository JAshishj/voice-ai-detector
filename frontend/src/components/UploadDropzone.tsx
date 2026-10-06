import { useRef, useState } from "react";

interface Props {
  onFile: (file: File) => void;
  disabled?: boolean;
}

const ACCEPT = ".mp3,.wav,.flac,audio/mpeg,audio/wav,audio/flac,audio/x-flac";

/** Drag-drop + browse upload zone. Keyboard operable via hidden input. */
export function UploadDropzone({ onFile, disabled }: Props) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  return (
    <div
      role="button"
      tabIndex={disabled ? -1 : 0}
      aria-label="Upload audio file"
      aria-disabled={disabled}
      onClick={() => !disabled && inputRef.current?.click()}
      onKeyDown={(e) => {
        if ((e.key === "Enter" || e.key === " ") && !disabled) {
          e.preventDefault();
          inputRef.current?.click();
        }
      }}
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        const f = e.dataTransfer.files?.[0];
        if (f && !disabled) onFile(f);
      }}
      className={`cursor-pointer rounded-[20px] border border-dashed bg-raised p-8 text-center transition-colors ${
        dragging ? "border-teal" : "border-hairline hover:border-teal/60"
      } ${disabled ? "cursor-not-allowed opacity-50" : ""}`}
    >
      <p className="font-display text-lg font-bold text-ink">
        Drop a voice clip here, or browse
      </p>
      <p className="mt-2 font-mono text-xs tracking-wider text-dim">
        MP3 · WAV · FLAC — UP TO ~30S
      </p>
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT}
        className="hidden"
        disabled={disabled}
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onFile(f);
          e.target.value = "";
        }}
      />
    </div>
  );
}
