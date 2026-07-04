<?php

namespace App\Helpers;

function format_date(string $label): string
{
    return sprintf('[%s]', $label);
}

function slugify(string $value, string $separator = '-'): string
{
    return strtolower(str_replace(' ', $separator, $value));
}
