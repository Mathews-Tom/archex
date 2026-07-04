<?php

namespace App\Models;

use App\Contracts\Arrayable;
use App\Traits\HasTimestamps;

class User implements Arrayable
{
    use HasTimestamps;

    public const MAX_NAME_LEN = 255;

    private int $id;
    protected string $email = '';

    public function __construct(private readonly string $name, int $id = 0)
    {
        $this->id = $id;
    }

    public static function create(string $name): self
    {
        return new self($name);
    }

    public function toArray(): array
    {
        return [
            'id' => $this->id,
            'name' => $this->name,
        ];
    }

    private function validate(): bool
    {
        return strlen($this->name) <= self::MAX_NAME_LEN;
    }
}
