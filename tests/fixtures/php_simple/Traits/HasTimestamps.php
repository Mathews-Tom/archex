<?php

namespace App\Traits;

trait HasTimestamps
{
    protected ?\DateTimeImmutable $createdAt = null;

    public function touch(): void
    {
        $this->createdAt = new \DateTimeImmutable();
    }

    public function createdAt(): ?\DateTimeImmutable
    {
        return $this->createdAt;
    }
}
