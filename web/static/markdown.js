/*
 * Minimal Markdown renderer for the project's own reports (reports/*.md).
 * Supports headings, paragraphs, bold/italic/code, links, images, nested
 * lists, tables, blockquotes and rules. Input is escaped before formatting.
 */
(function () {
  "use strict";

  const escapeHtml = (s) =>
    s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

  // Relative links in the reports point at repository files, which the web app
  // does not serve; keep their text and only link absolute URLs.
  function inline(text, opts) {
    const codes = [];
    let s = escapeHtml(text).replace(/`([^`]+)`/g, (_, c) => {
      codes.push(c);
      return `\u0000${codes.length - 1}\u0000`;
    });
    s = s.replace(/!\[([^\]]*)\]\(([^)\s]+)\)/g, (_, alt, src) =>
      `<img src="${opts.resolveImage(src)}" alt="${alt}" loading="lazy">`);
    s = s.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_, label, href) =>
      /^https?:\/\//.test(href) ? `<a href="${href}" target="_blank" rel="noopener">${label}</a>` : label);
    s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/(^|[^*\w])\*([^*\s][^*]*?)\*(?!\w)/g, "$1<em>$2</em>");
    s = s.replace(/(^|\W)_([^_\s][^_]*?)_(?!\w)/g, "$1<em>$2</em>");
    return s.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code>${codes[+i]}</code>`);
  }

  const LIST_RE = /^(\s*)([-*+]|\d+[.)])\s+(.*)$/;
  const isTableSep = (line) => /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/.test(line);
  const splitRow = (line) => line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());

  function renderList(lines, opts) {
    // Build nested lists from indentation
    const root = { children: [], indent: -1 };
    const stack = [root];
    let current = null;
    for (const line of lines) {
      const m = line.match(LIST_RE);
      if (!m) {
        if (current && line.trim()) current.text += " " + line.trim();
        continue;
      }
      const indent = m[1].replace(/\t/g, "    ").length;
      const ordered = /\d/.test(m[2]);
      while (stack.length > 1 && indent <= stack[stack.length - 1].indent) stack.pop();
      const parent = stack[stack.length - 1];
      let list = parent === root ? root.list : parent.item.sublist;
      if (!list) {
        list = { ordered, items: [] };
        if (parent === root) root.list = list; else parent.item.sublist = list;
      }
      current = { text: m[3], sublist: null };
      list.items.push(current);
      stack.push({ indent, item: current });
    }
    const toHtml = (list) => {
      const tag = list.ordered ? "ol" : "ul";
      return `<${tag}>${list.items.map((it) =>
        `<li>${inline(it.text, opts)}${it.sublist ? toHtml(it.sublist) : ""}</li>`).join("")}</${tag}>`;
    };
    return root.list ? toHtml(root.list) : "";
  }

  function render(md, opts = {}) {
    opts = { resolveImage: (src) => src, ...opts };
    const lines = md.replace(/\r\n?/g, "\n").split("\n");
    const out = [];
    let para = [];
    const flush = () => {
      if (para.length) out.push(`<p>${inline(para.join(" "), opts)}</p>`);
      para = [];
    };

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const trimmed = line.trim();

      if (!trimmed) { flush(); continue; }

      if (/^(-{3,}|\*{3,})$/.test(trimmed)) { flush(); out.push("<hr>"); continue; }

      const h = trimmed.match(/^(#{1,6})\s+(.*)$/);
      if (h) {
        flush();
        const level = h[1].length;
        out.push(`<h${level}>${inline(h[2], opts)}</h${level}>`);
        continue;
      }

      const img = trimmed.match(/^!\[([^\]]*)\]\(([^)\s]+)\)$/);
      if (img) {
        flush();
        out.push(`<figure><img src="${opts.resolveImage(img[2])}" alt="${escapeHtml(img[1])}" loading="lazy">` +
                 `<figcaption>${escapeHtml(img[1])}</figcaption></figure>`);
        continue;
      }

      if (trimmed.startsWith(">")) {
        flush();
        const block = [];
        while (i < lines.length && lines[i].trim().startsWith(">")) {
          block.push(lines[i].trim().replace(/^>\s?/, ""));
          i++;
        }
        i--;
        out.push(`<blockquote>${render(block.join("\n"), opts)}</blockquote>`);
        continue;
      }

      if (trimmed.startsWith("|") && i + 1 < lines.length && isTableSep(lines[i + 1])) {
        flush();
        const head = splitRow(line);
        const aligns = splitRow(lines[i + 1]).map((c) =>
          c.endsWith(":") ? (c.startsWith(":") ? "center" : "right") : "");
        i += 2;
        const rows = [];
        while (i < lines.length && lines[i].trim().startsWith("|")) rows.push(splitRow(lines[i++]));
        i--;
        const cell = (tag, c, j) =>
          `<${tag}${aligns[j] ? ` style="text-align:${aligns[j]}"` : ""}>${inline(c, opts)}</${tag}>`;
        out.push(`<div class="table-wrap"><table><thead><tr>${head.map((c, j) => cell("th", c, j)).join("")}</tr></thead>` +
                 `<tbody>${rows.map((r) => `<tr>${r.map((c, j) => cell("td", c, j)).join("")}</tr>`).join("")}</tbody></table></div>`);
        continue;
      }

      // A list may start right after a paragraph line (as in CommonMark)
      if (LIST_RE.test(line)) {
        flush();
        const block = [];
        while (i < lines.length) {
          const l = lines[i];
          if (LIST_RE.test(l) || (l.trim() && /^\s+/.test(l))) { block.push(l); i++; continue; }
          // A blank line inside a list continues it when an item or indented line follows
          if (!l.trim()) {
            let j = i + 1;
            while (j < lines.length && !lines[j].trim()) j++;
            if (j < lines.length && (/^\s+\S/.test(lines[j]) && LIST_RE.test(lines[j]))) { i = j; continue; }
          }
          break;
        }
        i--;
        out.push(renderList(block, opts));
        continue;
      }

      para.push(trimmed);
    }
    flush();
    return out.join("\n");
  }

  window.Markdown = { render, inline: (t, o = {}) => inline(t, { resolveImage: (s) => s, ...o }) };
})();
