"""haskala_grading — the grading pipeline shared by every haskala product.

    from haskala_grading import core, report

    pages, evaluation = core.check_pages(file_bytes_list, rubric_override=…,
                                         question=…, model_key=…)
    docx_bytes = report.build_evaluation_docx(evaluation, out_name, rubric_name,
                                              pages=pages)

Two modules, one contract between them: `core` produces an evaluation dict in
the shape of `core.EVAL_SCHEMA`, and `report` turns that dict plus the page list
into a Word document. `report` imports nothing from `core`, so it can be used on
its own and there is no import cycle.

WHERE THIS LIVES, AND WHY IT IS VENDORED
────────────────────────────────────────
Source of truth: haskala/lib/haskala_grading/. Products do not import it from
here at runtime — `gcloud run deploy --source .` uploads only the product
directory, so a sibling lib/ would simply be absent from the container. Each
product's deploy.sh therefore rsyncs this package into the product directory
first, and haskala/.gitignore keeps that vendored copy out of git.

The practical rule: **edit lib/, never a vendored copy.** A vendored
haskala_grading/ inside a product directory is build output — it is overwritten
on the next deploy and any change made there is lost silently.

WHY IT EXISTS
─────────────
The OCR prompt and the evaluation prompt had four independent copies across
products/checker, products/math-checker, products/scan2 and
products/oris-scanner, which had fully diverged (1,751 changed lines between
oris-scanner's core.py and scan2's alone), as had the two copies of rubrics/.
This package was extracted from oris-scanner — the most hardened of the four —
when products/form-checker would have become the fifth.

products/checker and products/math-checker have NOT been migrated. Both are
dormant, and moving them is a separate job with its own verification.
"""
