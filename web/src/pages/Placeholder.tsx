// Stand-in for pages not built yet: says what the page will do and which plan step builds it.
import { PageHeader, Panel } from "../components/ui";
import type { PageDef } from "../pages";

export default function Placeholder({ page }: { page: PageDef }) {
  return (
    <div className="flex flex-col gap-5">
      <PageHeader eyebrow={page.label} title={page.title}>
        {page.purpose}
      </PageHeader>
      <Panel title={`Arrives in ${page.arrives}`} sub="This page is not part of the submitted version.">
        <p className="m-0 text-muted">Use the navigation on the left.</p>
      </Panel>
    </div>
  );
}
