# fasteval Documentation

This folder contains the source documentation for fasteval. The documentation is rendered by the `fasteval-doc-plugin` project.

## Directory Structure

```
docs/
├── docs.config.json          # Main config: defines section order
├── README.md                 # This file
├── getting-started/          # Each folder is a section
│   ├── _meta.json            # Section metadata (title, icon, order)
│   ├── introduction.mdx      # Page files
│   ├── installation.mdx
│   └── quickstart.mdx
├── core-concepts/
│   ├── _meta.json
│   └── *.mdx
└── ...
```

## How Ordering Works

### Section Order

Sections are ordered in two places:

1. **`docs.config.json`** - Defines which sections exist and their display order:

```json
{
  "sections": [
    "getting-started",    // First section
    "core-concepts",      // Second section
    "llm-metrics",        // Third section
    ...
  ]
}
```

2. **`_meta.json`** in each section folder - Defines section metadata:

```json
{
  "title": "Getting Started",
  "icon": "rocket",
  "order": 1
}
```

Available icons: `rocket`, `book`, `robot`, `database`, `chart`, `message`, `settings`, `code`, `puzzle`, `sparkles`, `check`

### Page Order

Pages within a section are ordered by the `order` field in their frontmatter:

```yaml
---
title: Introduction
description: Learn about fasteval
order: 1
---
```

Lower numbers appear first. Pages without an `order` default to 999.

## Creating a New Page

1. Create an `.mdx` file in the appropriate section folder:

```bash
touch docs/core-concepts/new-feature.mdx
```

2. Add frontmatter at the top:

```yaml
---
title: New Feature
description: A brief description of this feature
order: 6
sidebar:
  - title: Overview
    anchor: overview
  - title: Usage
    anchor: usage
  - title: Examples
    anchor: examples
---
```

3. Write your content using Markdown:

```markdown
# New Feature

Introduction paragraph...

## Overview

Content here...

## Usage

```python
import fasteval as fe
# code example
```

## Examples

More content...
```

## Creating a New Section

1. Create a folder for the section:

```bash
mkdir docs/new-section
```

2. Create `_meta.json` with section metadata:

```json
{
  "title": "New Section",
  "icon": "sparkles",
  "order": 10
}
```

3. Add the section to `docs.config.json`:

```json
{
  "sections": [
    "getting-started",
    "core-concepts",
    "new-section",    // Add here
    "advanced"
  ]
}
```

4. Create at least one page in the section (see "Creating a New Page" above).

## Sidebar Sub-Items

Sub-items appear as nested navigation under a page. They link to heading anchors.

### Adding Sub-Items

Add a `sidebar` array to your frontmatter:

```yaml
---
title: My Page
order: 1
sidebar:
  - title: Quick Start
    anchor: quick-start
  - title: Configuration
    anchor: configuration
  - title: Best Practices
    anchor: best-practices
---
```

### Anchor Format

Anchors are auto-generated from h2/h3 headings:
- `## Quick Start` → `anchor: quick-start`
- `## CI/CD Mode` → `anchor: ci-cd-mode`
- `## The fe.score() Function` → `anchor: the-fescore-function`

**Rules:**
- Lowercase
- Spaces and special characters become hyphens
- Leading/trailing hyphens are removed

### Best Practices for Sub-Items

✓ **Good sub-items** (conceptual sections):
- "Quick Start"
- "Configuration"
- "Best Practices"
- "How It Works"

✗ **Avoid** (too technical):
- "Parameters"
- "Return Type"
- "Class Methods"

## Syncing with Doc Plugin

After making changes, copy the updated files to the doc-plugin:

```bash
# Copy entire docs folder
cp -r docs/* ../fasteval-doc-plugin/src/docs/

# Or copy specific files
cp docs/getting-started/new-page.mdx ../fasteval-doc-plugin/src/docs/getting-started/

# Regenerate manifest (in doc-plugin directory)
cd ../fasteval-doc-plugin
node scripts/generate-manifest.js
```

## File Format Reference

### docs.config.json

```json
{
  "title": "fasteval Documentation",
  "description": "Decorator-first LLM evaluation for Python",
  "sections": [
    "getting-started",
    "core-concepts",
    "llm-metrics",
    "rag-metrics",
    "tool-tranjectory-metrics",
    "deterministic-metrics",
    "conversation-metrics",
    "human-review",
    "advanced",
    "api-reference"
  ]
}
```

### _meta.json (Section Metadata)

```json
{
  "title": "Section Title",
  "icon": "icon-name",
  "order": 1
}
```

### MDX Page Frontmatter

```yaml
---
title: Page Title
description: Brief description for SEO
order: 1
sidebar:
  - title: Sub-item Title
    anchor: heading-anchor
---
```

## Writing Style Guide

1. **Use clear, concise language** - Write for developers who want quick answers

2. **Start with examples** - Show code before explaining theory

3. **Use practical code snippets** - Real-world examples, not abstract ones

4. **Include expected output** - Show what users will see

5. **Link related concepts** - Help users discover more features

## Common Tasks

### Reorder a Section

1. Update `order` in the section's `_meta.json`
2. Update the array order in `docs.config.json`
3. Regenerate manifest

### Rename a Page

1. Update the `title` in frontmatter
2. The file name doesn't affect the display title

### Move a Page to Another Section

1. Move the `.mdx` file to the new section folder
2. Update the `order` in frontmatter to fit the new section
3. Regenerate manifest

### Add an Icon to a Section

Update `_meta.json`:

```json
{
  "title": "My Section",
  "icon": "sparkles",  // Add or change this
  "order": 5
}
```

## Troubleshooting

### Page not showing in sidebar

1. Check that the file has `.mdx` extension
2. Verify frontmatter has valid YAML (no syntax errors)
3. Ensure the section folder is listed in `docs.config.json`
4. Regenerate the manifest

### Sub-items not appearing

1. Verify `sidebar` array syntax in frontmatter
2. Check that anchors match your h2/h3 headings exactly
3. Ensure the doc-plugin manifest is updated

### Order not working

1. Check `order` is a number, not a string (`order: 1` not `order: "1"`)
2. Verify both `_meta.json` and page frontmatter have correct `order`
3. Regenerate the manifest
