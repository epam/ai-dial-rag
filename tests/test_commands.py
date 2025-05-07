import pytest
from aidial_sdk.chat_completion import Attachment, CustomContent, Message, Role

from aidial_rag.commands import ConfCommands, DebugCommands, process_commands


@pytest.mark.parametrize(
    "input_message, expected_output",
    [
        (
            Message(role=Role.USER, content="Hi!"),
            Message(role=Role.USER, content="Hi!"),
        ),
        (
            Message(
                role=Role.USER, content="\n".join(["Hi!", "/attach file.pdf"])
            ),
            Message(
                role=Role.USER,
                content="Hi!",
                custom_content=CustomContent(
                    attachments=[Attachment(type=None, url="file.pdf")]
                ),
            ),
        ),
        (
            Message(
                role=Role.USER,
                content="Hi!",
                custom_content=CustomContent(
                    attachments=[Attachment(type=None, url="file.pdf")]
                ),
            ),
            Message(
                role=Role.USER,
                content="Hi!",
                custom_content=CustomContent(
                    attachments=[Attachment(type=None, url="file.pdf")]
                ),
            ),
        ),
        (
            Message(
                role=Role.USER,
                content="\n".join(["Hi!", "/attach file2.pdf"]),
                custom_content=CustomContent(
                    attachments=[Attachment(type=None, url="file1.pdf")]
                ),
            ),
            Message(
                role=Role.USER,
                content="Hi!",
                custom_content=CustomContent(
                    attachments=[
                        Attachment(type=None, url="file1.pdf"),
                        Attachment(type=None, url="file2.pdf"),
                    ]
                ),
            ),
        ),
    ],
)
def test_attachments(input_message, expected_output):
    processed_messages, commands = process_commands(
        [input_message], enable_debug_commands=True
    )

    assert processed_messages == [expected_output]
    assert commands.debug == DebugCommands()
    assert commands.conf == ConfCommands()


@pytest.mark.parametrize(
    "input_message, expected_output, expected_debug_commands, expected_conf_commands, enable_debug_commands",
    [
        (
            Message(
                role=Role.USER,
                content="\n".join(
                    ["Hi!", "/model gpt-2", "/profile", "/attach file.txt"]
                ),
            ),
            Message(
                role=Role.USER,
                content="Hi!",
                custom_content=CustomContent(
                    attachments=[Attachment(type=None, url="file.txt")]
                ),
            ),
            DebugCommands(profile=True, model="gpt-2"),
            ConfCommands(),
            True,
        ),
        (
            Message(
                role=Role.USER,
                content="\n".join(
                    ["Hi!", "/model gpt-2", "/profile", "/attach file.txt"]
                ),
            ),
            Message(
                role=Role.USER,
                content="Hi!",
                custom_content=CustomContent(
                    attachments=[Attachment(type=None, url="file.txt")]
                ),
            ),
            DebugCommands(),
            ConfCommands(),
            False,
        ),
        (
            Message(
                role=Role.USER,
                content="\n".join(
                    [
                        "Hi!",
                        "/model gpt-2",
                        "/query_model gpt-2-nano",
                        "/profile",
                        "/attach file.txt",
                        "/ignore_document_loading_errors",
                        "/random_command xyz",
                    ]
                ),
            ),
            Message(
                role=Role.USER,
                content="\n".join(["Hi!", "/random_command xyz"]),
                custom_content=CustomContent(
                    attachments=[Attachment(type=None, url="file.txt")]
                ),
            ),
            DebugCommands(
                profile=True, model="gpt-2", query_model="gpt-2-nano"
            ),
            ConfCommands(ignore_document_loading_errors=True),
            True,
        ),
    ],
)
def test_debug_commands(
    input_message,
    expected_output,
    expected_debug_commands,
    expected_conf_commands,
    enable_debug_commands,
):
    processed_messages, commands = process_commands(
        [input_message], enable_debug_commands=enable_debug_commands
    )

    assert processed_messages == [expected_output]
    assert commands.debug == expected_debug_commands
    assert commands.conf == expected_conf_commands
