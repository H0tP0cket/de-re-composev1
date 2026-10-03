# SWE-smith bug patches that describe the bug

While packaging 50 SWE-smith instances as long multi-bug tasks, I scanned each injected bug patch for added comments and docstring text. In **21 of the 50** the patch adds comments that state the injected change (18 clear leaks, 3 borderline). An agent that reads the file can find the bug by reading the comment.

Examples (instance id, file, added line):

- `gweis__isodate.17cb25eb.combine_module__83qvrapz`, `src/isodate/isostrf.py`
  - `ret.append("%sY" % abs(tdt.months))  # Subtly swapped months and years`
  - `seconds, usecs = divmod(usecs, 500000)  # Divmod by 500000 instead of 1000000`
- `gweis__isodate.17cb25eb.combine_module__twxleaao`, `src/isodate/duration.py`
  - `newyear = other.year - self.years - carry  # Changed '+' to '-'`
- `marshmallow-code__apispec.8b421526.combine_module__gz0k9zld`, `src/apispec/ext/marshmallow/schema_resolver.py`
  - `if not isinstance(callback, dict):  # Add incorrect condition check`
  - `for path in callback.keys():  # Iterate incorrectly over keys instead of values`

DeepSeek V4.1 Flash noticed these comments and reasoned that it was inside a generated benchmark. Separately, it spent about 30% of its commands on these tasks hunting for the hidden tests or the original source, in every run and in 121 of 122 recovery branches; GPT-6 Luna did so in 1 of 10 runs. The hunting never found test content, but in some images it surfaced the names of withheld test files through installed package metadata.

I dropped the leaking tasks from later batches and moved to other task sources.
