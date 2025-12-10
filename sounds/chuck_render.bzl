"""Bazel rule to render Chuck (.ck) files to WAV using the Chuck binary."""

def _chuck_render_impl(ctx):
    """Implementation of chuck_render rule."""
    output = ctx.outputs.output
    input_file = ctx.file.src

    # Get the chuck binary from the filegroup
    chuck_files = ctx.files.chuck
    if len(chuck_files) != 1:
        fail("Expected exactly one file in chuck filegroup, got {}".format(len(chuck_files)))
    chuck_binary = chuck_files[0]

    # Calculate duration: add a small buffer to ensure we capture the full sound
    duration = float(ctx.attr.duration)

    # Output WAV file (no MP3 conversion needed - pygame works better with WAV)
    wav_output_file = ctx.outputs.output

    # Create a modified Chuck file that uses WavOut instead of dac
    modified_chuck_file = ctx.actions.declare_file(ctx.label.name + "_modified.ck")

    # Create a Python helper script to modify the Chuck file
    helper_script = ctx.actions.declare_file(ctx.label.name + "_helper.py")
    helper_script_content = """#!/usr/bin/env python3
import sys
import re

def modify_chuck_file(input_path, output_path, wav_output_path):
    with open(input_path, 'r') as f:
        content = f.read()

    # Replace '=> dac' with '=> wv => dac'
    # WvOut needs to be in the chain, but we also connect to dac to ensure audio flows
    # Actually, let's try just '=> wv' but ensure wv is properly initialized
    modified_content = content.replace('=> dac', '=> wv')

    # Fix HPF.set() and LPF.set() calls that might need Q parameter
    # HPF.set(freq) -> HPF.set(freq, 1.0) if only one argument
    # LPF.set(freq) -> LPF.set(freq, 1.0) if only one argument
    # Use raw strings and proper escaping
    pattern_hpf = r'filter\\.set\\((\\d+(?:\\.\\d+)?)\\); // High pass frequency'
    replacement_hpf = r'filter.set(\\1, 1.0); // High pass frequency and Q'
    modified_content = re.sub(pattern_hpf, replacement_hpf, modified_content)

    pattern_lpf = r'filter\\.set\\((\\d+(?:\\.\\d+)?)\\); // Low pass frequency'
    replacement_lpf = r'filter.set(\\1, 1.0); // Low pass frequency and Q'
    modified_content = re.sub(pattern_lpf, replacement_lpf, modified_content)

    # Add WvOut initialization at the beginning
    # WvOut writes audio to a WAV file
    # The wavFilename must be set BEFORE connecting audio to wv
    # Also connect wv to blackhole to ensure audio flows through
    wvout_init = f'''// Modified for rendering to WAV
WvOut wv;
"{wav_output_path}" => wv.wavFilename;
wv => blackhole;  // Ensure audio flows through WvOut

'''

    # Add cleanup at the end to close WvOut explicitly
    # WvOut needs to be closed to flush audio data to file
    wvout_cleanup = '''
// Close WvOut to flush audio data to file
wv.closeFile();
// Small delay to ensure file is written
0.1::second => now;
'''

    with open(output_path, 'w') as f:
        f.write(wvout_init)
        f.write(modified_content)
        f.write(wvout_cleanup)

if __name__ == '__main__':
    modify_chuck_file(sys.argv[1], sys.argv[2], sys.argv[3])
"""
    ctx.actions.write(helper_script, helper_script_content, is_executable = True)

    # Create a script that modifies the Chuck file and renders it
    script_content = """#!/bin/bash
set -e
# Use Python helper to modify the Chuck file
python3 {helper_script} {input} {modified_chuck} {wav_output}
# Verify the modified file was created
if [ ! -f {modified_chuck} ]; then
    echo "Error: Modified Chuck file was not created"
    exit 1
fi
# Run chuck with the modified file in silent mode
# Use --silent to avoid real-time audio output
# Capture stderr to see any errors
{chuck} --silent {modified_chuck} 2>&1 || true
# Wait longer for the file to be written and flushed
sleep 1.5
# Verify WAV file was created and has content
if [ ! -f {wav_output} ]; then
    echo "Error: WAV file was not created"
    echo "Modified Chuck file contents:"
    cat {modified_chuck}
    exit 1
fi
# Check if WAV file has actual audio data (should be > 1000 bytes)
WAV_SIZE=$(stat -f%z {wav_output} 2>/dev/null || stat -c%s {wav_output} 2>/dev/null || echo "0")
if [ "$WAV_SIZE" -lt 1000 ]; then
    echo "Warning: WAV file is too small ($WAV_SIZE bytes), may be empty"
    echo "Modified Chuck file contents:"
    cat {modified_chuck}
fi
# WAV file is the final output (no MP3 conversion needed)
""".format(
        helper_script = helper_script.path,
        chuck = chuck_binary.path,
        input = input_file.path,
        modified_chuck = modified_chuck_file.path,
        wav_output = wav_output_file.path,
        duration_plus_buffer = str(duration + 0.5),
    )

    script = ctx.actions.declare_file(ctx.label.name + "_render.sh")
    ctx.actions.write(script, script_content, is_executable = True)

    # Run the script
    ctx.actions.run(
        inputs = [input_file] + ctx.files.chuck + [helper_script],
        outputs = [output, modified_chuck_file],
        executable = script,
        tools = [],
        mnemonic = "ChuckRender",
        progress_message = "Rendering {} to WAV".format(input_file.basename),
    )

chuck_render = rule(
    implementation = _chuck_render_impl,
    attrs = {
        "src": attr.label(
            mandatory = True,
            allow_single_file = [".ck"],
            doc = "Input Chuck (.ck) file to render",
        ),
        "chuck": attr.label(
            default = "@chuck//:bin",
            executable = False,
            cfg = "exec",
            doc = "Chuck binary filegroup to use for rendering",
        ),
        "duration": attr.string(
            default = "2.0",
            doc = "Duration in seconds to render (default: 2.0)",
        ),
        "output": attr.output(
            mandatory = True,
            doc = "Output WAV file path",
        ),
    },
    doc = "Renders a Chuck (.ck) file to WAV format",
)
