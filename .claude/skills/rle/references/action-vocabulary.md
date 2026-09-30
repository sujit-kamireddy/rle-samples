## The action vocabulary: `GET /schema` is the whole story

RLE derives everything the model may do from the `action` schema your OpenEnv server publishes at
`GET /schema`. There is no separate tool spec to write, register, or keep in sync. The rule:

> Of all the action variants in the schema, **the one variant declaring `model_response_field` receives
> the model's completion text. Every other variant is offered to the model as a tool named after its
> discriminator.**

### Case 1 — no tools (single action)

A plain `Action` subclass. `math_rl` (`server/schema.py`):

```python
class MathAction(Action):
    answer_text: str = Field(default="", description="...")
```

With `model_response_field = "answer_text"` there is one variant, it carries the response field,
and the model is given no tools. The completion text arrives in `action.answer_text`.

### Case 2 — with tools (discriminated union)

A `RootModel` union with an explicit discriminator. `code_rl` (`server/schema.py`):

```python
class CheckSolutionAction(Action):
    type: Literal["check_solution"] = "check_solution"
    code: str = Field(description="Python code implementing the solution.")

class SubmitAnswerAction(Action):
    type: Literal["submit_answer"] = "submit_answer"
    code_text: str = Field(default="", description="...")

class CodeAction(RootModel[Annotated[
    Union[CheckSolutionAction, SubmitAnswerAction],
    Field(discriminator="type"),
]]):
    ...
```

The `RootModel` union is what makes `GET /schema`'s `action` a `oneOf` with a `discriminator`, which
is how RLE tells variants apart. With `model_response_field = "code_text"`, `SubmitAnswerAction` is
the response variant and `check_solution` becomes a tool — its name is the discriminator value, its
parameters are the variant's other properties, and its description is the class docstring. Reach the
concrete action through `.root` in `step()`.

**Adding a tool is therefore: add a union member.** Nothing else.

### Conformance rules RLE enforces

The rollout fails with a conformance error — before any model call — if:

| Condition | Why |
| --- | --- |
| `action` is not a JSON object | nothing to read |
| the schema declares no variant | nothing the model could emit |
| **no** variant declares `model_response_field` | the error lists every property it *did* find — read it, it is usually a typo or the wrong field name in `rle.toml` |
| **more than one** variant declares `model_response_field` | RLE cannot tell which one carries the graded response |
| a non-response variant has no discriminator | it cannot be named as a tool |
| two variants share a discriminator value | ambiguous tool name |

### Authoring constraints that follow from this

- **Do not put the response field on more than one variant.** If two actions both need free text,
  give them different property names and pick one as the response field.
- **Keep `$ref` chains shallow.** `$ref` resolution is bounded at 8 hops (deliberately, so a `$ref`
  cycle cannot spin), which no legitimate action schema approaches — but deeply nested generic
  models can surprise you.
- **`Optional[...]` is fine.** Null union branches are handled; the non-null branch is what
  describes the parameter.
- **Descriptions are prompt text.** The model sees your `Field(description=...)` and class
  docstrings. Vague descriptions produce vague policies.
- **Do not add an episode-identity field** (`problem_id`, `instance_id`, …) to the action. RLE drives
  Gym/OpenEnv only over `/ws`, one environment instance per connection, so `step()` resolves the
  episode from what `reset()` stored on `self`. An identity field on the action is a field the model
  can get wrong.
