PAGE_DESCRIPTION_PROMPT_TEMPLATE = """
Please create detailed description of provided image.
Ignore page header, footer, basic logo and background.
Describe all images (illustration), tables.
Text with bullet points is NOT a table or image.

Use only provided information.
DO NOT make up answer.

Provide answer in JSON format with fields:
{{
    "page_summary": "page summary here",
    "keyfact"     : "the most important fact from the image",
    "image_quality": {{
        "level": "level of image quality (normal, detailed)", 
        "explanation": "explain why this detailisation is required"
    }},
    "images":[
        {{
            "description": "image description",
            "type"       : "image type (photo, illustration, diagram, etc.)",
            "keyfact"    : "the most important fact from the image"
        }}
    ],
    "tables":[
        {{
            "description": "table description",
            "keyfact"    : "the most important fact from the table"
        }}
    ]
}}
"""  # noqa: W291
