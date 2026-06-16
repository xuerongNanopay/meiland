pub fn replace_python_style_get(template: &str) -> String {
    let mut template = template.to_owned();

    // Special case:
    // message.get('reasoning') or message.get('reasoning_content')
    template = template.replace(
        "message.get('reasoning') or message.get('reasoning_content')",
        "message.reasoning|default(message.reasoning_content|default(none))",
    );

    // List-like fields should default to []
    for field in ["tool_calls", "tool_responses"] {
        template = template.replace(
            &format!("message.get('{field}')"),
            &format!("message.{field}|default([])"),
        );
    }

    // Generic fallback:
    // obj.get('field') -> obj.field|default(none)
    let re = regex::Regex::new(
        r"([a-zA-Z_][a-zA-Z0-9_]*)\.get\('([a-zA-Z_][a-zA-Z0-9_]*)'\)"
    ).unwrap();

    re.replace_all(&template, "$1.$2|default(none)")
        .to_string()
}