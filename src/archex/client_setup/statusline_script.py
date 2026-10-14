"""Claude Code status-line script rendering."""

from __future__ import annotations

from archex.project import PROJECT_DIR_NAME
from archex.status_snapshot import (
    DEFAULT_STALE_AFTER_SECONDS,
    SNAPSHOT_FILENAME,
    STATUS_SNAPSHOT_VERSION,
    WATCH_OBSERVATION_TTL_SECONDS,
)


def render_statusline_script() -> str:
    """Render the status-line script with the snapshot contract baked in."""
    return (
        _STATUSLINE_SCRIPT_TEMPLATE.replace(
            "__ARCHEX_SNAPSHOT_VERSION__", str(STATUS_SNAPSHOT_VERSION)
        )
        .replace("__ARCHEX_STALE_DEFAULT__", str(DEFAULT_STALE_AFTER_SECONDS))
        .replace("__ARCHEX_WATCH_TTL__", str(WATCH_OBSERVATION_TTL_SECONDS))
        .replace("__ARCHEX_SNAPSHOT_RELATIVE_PATH__", f"{PROJECT_DIR_NAME}/{SNAPSHOT_FILENAME}")
    )


_STATUSLINE_SCRIPT_TEMPLATE = r"""#!/bin/sh
# archex:statusline - Claude Code status line renderer (R23).
#
# Installed by `archex install-client claude-code --statusline`, removed by
# `--remove-statusline`. Renders the bounded status snapshot archex publishes
# at <repo>/__ARCHEX_SNAPSHOT_RELATIVE_PATH__.
#
# Claude Code re-runs this command on every repaint (debounced at 300ms, and
# it cancels an in-flight script when a new update arrives), so the renderer
# uses shell builtins exclusively: no `jq`, no `python`, no `date`, no
# `archex`, and no command substitution anywhere. It therefore forks no
# process, opens no index, and parses no source. The test suite proves that
# by running it with an empty PATH.
#
# Snapshot version this renderer understands: __ARCHEX_SNAPSHOT_VERSION__.
# A different version prints `unsupported` rather than guessing at fields.

set -u

archex_version_supported=__ARCHEX_SNAPSHOT_VERSION__
archex_stale_default=__ARCHEX_STALE_DEFAULT__
archex_watch_ttl=__ARCHEX_WATCH_TTL__

# --- Locate the snapshot ----------------------------------------------------
# Claude Code writes one line of session JSON to stdin, carrying the session
# directory as "cwd". Extracting it by parameter expansion avoids a JSON
# parser; an unreadable payload falls back to the process working directory,
# and the upward walk then handles being started in a subdirectory.
payload=""
IFS= read -r payload 2>/dev/null || :

archex_dir=""
case $payload in
*'"cwd":'*)
	archex_rest=${payload#*'"cwd":'}
	while :; do
		case $archex_rest in
		' '*) archex_rest=${archex_rest# } ;;
		*) break ;;
		esac
	done
	case $archex_rest in
	'"'*)
		archex_rest=${archex_rest#\"}
		archex_dir=${archex_rest%%\"*}
		;;
	esac
	;;
esac
# Only an absolute, existing directory is trusted. A payload whose earlier
# values happen to contain the literal `"cwd":` text, or a relative or
# nonexistent path, falls back to the process working directory rather than
# steering the walk below somewhere else.
case $archex_dir in
/*) ;;
*) archex_dir="" ;;
esac
if [ -z "$archex_dir" ] || [ ! -d "$archex_dir" ]; then
	archex_dir=$PWD
fi

archex_snapshot=${ARCHEX_STATUS_SNAPSHOT:-}
if [ -z "$archex_snapshot" ]; then
	archex_probe=$archex_dir
	while [ -n "$archex_probe" ]; do
		if [ -f "$archex_probe/__ARCHEX_SNAPSHOT_RELATIVE_PATH__" ]; then
			archex_snapshot="$archex_probe/__ARCHEX_SNAPSHOT_RELATIVE_PATH__"
			break
		fi
		case $archex_probe in
		*/?*) archex_probe=${archex_probe%/*} ;;
		*) archex_probe="" ;;
		esac
	done
fi

if [ -z "$archex_snapshot" ] || [ ! -f "$archex_snapshot" ]; then
	printf 'archex missing - no status snapshot - run: archex index\n'
	exit 0
fi

# --- Read it ----------------------------------------------------------------
# The document is written with sorted keys and two-space indentation, so every
# scalar occupies one line as `  "key": value,`. Splitting that with
# parameter expansion is exact for this writer and degrades to an empty field
# (hence `corrupt`) for anything else.
archex_field_version=""
archex_field_state=""
archex_field_revision=""
archex_field_files=0
archex_field_chunks=0
archex_field_pending=0
archex_field_pending_complete=""
archex_field_reindex=""
archex_field_epoch=0
archex_field_watch_epoch=0

archex_line=""
while IFS= read -r archex_line || [ -n "$archex_line" ]; do
	case $archex_line in
	*'": '*) ;;
	*) continue ;;
	esac
	archex_key=${archex_line#*\"}
	archex_key=${archex_key%%\"*}
	archex_value=${archex_line#*\": }
	archex_value=${archex_value%,}
	case $archex_value in
	\"*\")
		archex_value=${archex_value#\"}
		archex_value=${archex_value%\"}
		;;
	esac
	case $archex_key in
	version) archex_field_version=$archex_value ;;
	state) archex_field_state=$archex_value ;;
	index_revision) archex_field_revision=$archex_value ;;
	files_indexed) archex_field_files=$archex_value ;;
	chunks_indexed) archex_field_chunks=$archex_value ;;
	pending_delta_files) archex_field_pending=$archex_value ;;
	pending_view_complete) archex_field_pending_complete=$archex_value ;;
	reindex_required) archex_field_reindex=$archex_value ;;
	written_at_epoch) archex_field_epoch=$archex_value ;;
	watch_observed_epoch) archex_field_watch_epoch=$archex_value ;;
	esac
done < "$archex_snapshot"

# Numeric fields are normalized before any arithmetic. Non-digits become 0,
# and a leading zero is stripped: POSIX arithmetic reads `08` as octal and
# aborts the whole script on the invalid digit, which would replace a
# `corrupt` line with an error on stderr and no status at all.
for archex_numeric in files chunks pending epoch watch_epoch; do
	case $archex_numeric in
	files) archex_probe_value=$archex_field_files ;;
	chunks) archex_probe_value=$archex_field_chunks ;;
	pending) archex_probe_value=$archex_field_pending ;;
	epoch) archex_probe_value=$archex_field_epoch ;;
	*) archex_probe_value=$archex_field_watch_epoch ;;
	esac
	case $archex_probe_value in
	'' | *[!0-9]*) archex_probe_value=0 ;;
	*)
		while :; do
			case $archex_probe_value in
			0?*) archex_probe_value=${archex_probe_value#0} ;;
			*) break ;;
			esac
		done
		;;
	esac
	case $archex_numeric in
	files) archex_field_files=$archex_probe_value ;;
	chunks) archex_field_chunks=$archex_probe_value ;;
	pending) archex_field_pending=$archex_probe_value ;;
	epoch) archex_field_epoch=$archex_probe_value ;;
	*) archex_field_watch_epoch=$archex_probe_value ;;
	esac
done

# --- Classify ---------------------------------------------------------------
# `corrupt` and `unsupported` are deliberately separate: a document that is
# unparsable or carries no usable version needs re-publishing, while a
# document from a schema this build does not read needs an archex upgrade.
# The Python reader and the omp/Pi module classify identically.
case $archex_field_version in
'' | *[!0-9]*)
	printf 'archex corrupt - unreadable snapshot - run: archex status\n'
	exit 0
	;;
esac
if [ "$archex_field_version" != "$archex_version_supported" ]; then
	printf 'archex unsupported - snapshot v%s - upgrade archex\n' "$archex_field_version"
	exit 0
fi
case $archex_field_state in
fresh | dirty | pending) ;;
*)
	printf 'archex corrupt - unreadable snapshot - run: archex status\n'
	exit 0
	;;
esac

# `EPOCHSECONDS` is a shell builtin variable in bash 5+ and zsh, and absent in
# dash and bash 3.2 (macOS `/bin/sh`). Where it is absent this renderer has no
# clock it can read without forking `date`, so it reports the measured state
# and its measurement time and leaves the `stale` judgement to
# `archex status --cached`, which has a clock. Paying a fork on every repaint
# to gain one label is the wrong trade.
if [ -n "${ZSH_VERSION:-}" ]; then
	# `zmodload` is a zsh builtin, so this costs no process. Guarded on
	# ZSH_VERSION because under sh/dash the word would be an external
	# command, which is exactly what this renderer refuses to launch.
	zmodload zsh/datetime 2>/dev/null || :
fi
archex_now=${EPOCHSECONDS:-}
case $archex_now in
'' | *[!0-9]*) archex_now="" ;;
esac

archex_age=""
if [ -n "$archex_now" ] && [ "$archex_field_epoch" -gt 0 ]; then
	archex_age=$((archex_now - archex_field_epoch))
	if [ "$archex_age" -lt 0 ]; then
		archex_age=0
	fi
fi

archex_budget=${ARCHEX_STATUS_STALE_AFTER_SECONDS:-$archex_stale_default}
case $archex_budget in
'' | *[!0-9]*) archex_budget=$archex_stale_default ;;
esac
if [ -n "$archex_age" ] && [ "$archex_age" -gt "$archex_budget" ]; then
	archex_field_state=stale
fi

archex_watch=""
if [ -n "$archex_now" ] && [ "$archex_field_watch_epoch" -gt 0 ]; then
	if [ $((archex_now - archex_field_watch_epoch)) -le "$archex_watch_ttl" ]; then
		archex_watch=" - watch"
	fi
fi

# --- Render -----------------------------------------------------------------
archex_detail=""
case $archex_field_state in
fresh) archex_detail=" - $archex_field_files files, $archex_field_chunks chunks" ;;
pending)
	# The model's field defaults to true, so an absent field means a complete
	# view here too; only an explicit `false` marks the count as a lower bound.
	if [ "$archex_field_pending_complete" = "false" ]; then
		archex_detail=" - $archex_field_pending+ awaiting sync"
	else
		archex_detail=" - $archex_field_pending awaiting sync"
	fi
	;;
dirty)
	if [ "$archex_field_reindex" = "true" ]; then
		archex_detail=" - reindex required"
	else
		archex_detail=" - index behind tree"
	fi
	;;
stale) archex_detail=" - unverified since measurement" ;;
esac

archex_revision_segment=""
if [ -n "$archex_field_revision" ]; then
	archex_revision_short=$archex_field_revision
	# Trim to the first eight characters, but only when there are more than
	# eight: for a shorter value the `????????` pattern does not match and the
	# suffix trim would collapse the whole string to empty.
	case $archex_field_revision in
	?????????*)
		archex_revision_tail=${archex_field_revision#????????}
		archex_revision_short=${archex_field_revision%"$archex_revision_tail"}
		;;
	esac
	archex_revision_segment=" - rev $archex_revision_short"
fi

archex_age_segment=""
if [ -n "$archex_age" ]; then
	if [ "$archex_age" -lt 60 ]; then
		archex_age_segment=" - ${archex_age}s ago"
	elif [ "$archex_age" -lt 3600 ]; then
		archex_age_segment=" - $((archex_age / 60))m ago"
	else
		archex_age_segment=" - $((archex_age / 3600))h ago"
	fi
fi

printf 'archex %s%s%s%s%s\n' \
	"$archex_field_state" \
	"$archex_detail" \
	"$archex_revision_segment" \
	"$archex_age_segment" \
	"$archex_watch"
"""
