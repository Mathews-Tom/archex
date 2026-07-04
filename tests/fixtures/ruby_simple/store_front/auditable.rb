# frozen_string_literal: true

module StoreFront
  module Auditable
    def audit!(event)
      @events ||= []
      @events << event
    end
  end
end
