# frozen_string_literal: true

require "set"
require_relative "../models/user"

module StoreFront
  module Services
    class UserService
      BATCH_SIZE = 25

      def initialize(users = Set.new)
        @users = users
      end

      def register(email)
        user = Models::User.new(email: email)
        @users.add(user)
        user
      end

      def serialize(user)
        {
          email: user.email,
          role: user.role,
          name: user.display_name
        }
      end
    end
  end
end
