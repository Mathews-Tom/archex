# frozen_string_literal: true

require_relative "../store_front/auditable"
require_relative "../support/slugger"

module StoreFront
  module Models
    class User < ApplicationRecord
      include StoreFront::Auditable
      extend StoreFront::Slugger

      DEFAULT_ROLE = "customer"

      attr_reader :email, :role

      def initialize(email:, role: DEFAULT_ROLE)
        @email = email
        @role = role
      end

      def self.find_by_email(email)
        new(email: email)
      end

      def display_name
        email.split("@").first
      end

      protected

      def normalized_role
        role.to_s.downcase
      end

      private

      def normalize_email
        email.to_s.strip.downcase
      end
    end
  end
end
