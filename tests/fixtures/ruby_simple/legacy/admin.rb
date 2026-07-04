# frozen_string_literal: true

module StoreFront
  module Legacy
    class Admin
      def initialize(name)
        @name = name
      end
    end

    class Guest
      def initialize
        @name = "guest"
      end
    end
  end
end
