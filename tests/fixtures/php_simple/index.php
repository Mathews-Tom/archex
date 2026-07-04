<?php

namespace App;

use App\Models\Status;
use App\Models\User;
use App\Services\UserService;

require_once __DIR__ . '/vendor/autoload.php';

$user = User::create('Ada');
$service = UserService::make($user);

echo $service->summarize(Status::Active, $user);
