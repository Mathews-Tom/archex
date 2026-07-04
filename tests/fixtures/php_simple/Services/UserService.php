<?php

namespace App\Services;

use App\Contracts\Arrayable as Arr;
use App\Models\{Status, User};
use function App\Helpers\format_date;

class UserService
{
    private const DEFAULT_LIMIT = 10;

    private static array $registry = [];

    public function __construct(private readonly User $user)
    {
    }

    public static function make(User $user): self
    {
        return new self($user);
    }

    public function summarize(Status $status, Arr $item): string
    {
        return format_date($status->label()) . get_class($item);
    }

    protected function limit(): int
    {
        return self::DEFAULT_LIMIT;
    }
}
