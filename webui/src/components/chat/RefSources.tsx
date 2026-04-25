import type { RagContextItem } from "../../types/chat";
import { sourceLabel } from "../../lib/chat/utils";

type RefSourcesProps = {
  refs: RagContextItem[];
};

export function RefSources({ refs }: RefSourcesProps) {
  return (
    <div className="self-stretch max-w-full rounded-lg border border-[#e5e5e5] bg-[#fafafa] p-4">
      <h2 className="text-sm font-medium text-[#666] mb-3 m-0">参考来源</h2>
      <ul className="space-y-4 list-none m-0 p-0">
        {refs.map((item, i) => {
          const meta = item.metadata ?? {};
          const src = sourceLabel(meta.source);
          const page =
            meta.page != null && meta.page !== ""
              ? `第 ${String(meta.page)} 页`
              : null;
          const isImage = item.type === "image";
          return (
            <li
              key={`${i}-${src}-${page ?? ""}`}
              className="rounded-md border border-[#e5e5e5] bg-white p-3 text-sm"
            >
              <div className="flex flex-wrap items-center gap-2 text-[#888] text-xs mb-2">
                <span className="text-[#666] font-medium">[{i + 1}]</span>
                <span
                  className={
                    isImage
                      ? "rounded bg-violet-100 text-violet-700 px-1.5 py-0.5"
                      : "rounded bg-gray-100 text-gray-700 px-1.5 py-0.5"
                  }
                >
                  {isImage ? "图片" : "文本"}
                </span>
                <span
                  className="text-[#555] truncate max-w-[min(100%,18rem)]"
                  title={
                    typeof meta.source === "string" ? meta.source : undefined
                  }
                >
                  {src}
                </span>
                {page && <span>{page}</span>}
              </div>
              {isImage && item.image_data ? (
                <div className="mt-2">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={item.image_data}
                    alt={`引用图 ${i + 1}`}
                    className="max-w-full max-h-64 rounded border border-[#e5e5e5] object-contain bg-[#f5f5f5]"
                  />
                </div>
              ) : null}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
