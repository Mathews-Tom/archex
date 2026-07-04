<?php

namespace App\Legacy {
    interface Loggable
    {
        public function log(string $message): void;
    }

    final class FileLogger implements Loggable
    {
        private string $path;

        public function __construct(string $path)
        {
            $this->path = $path;
        }

        public function log(string $message): void
        {
            file_put_contents($this->path, $message . PHP_EOL, FILE_APPEND);
        }
    }
}
