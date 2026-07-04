# frozen_string_literal: true

require "json"
require_relative "store_front/auditable"
require_relative "support/slugger"
require_relative "models/user"
require_relative "services/user_service"

service = StoreFront::Services::UserService.new
puts JSON.generate(service.serialize(StoreFront::Models::User.find_by_email("ada@example.com")))
