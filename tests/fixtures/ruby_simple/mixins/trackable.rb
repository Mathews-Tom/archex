# frozen_string_literal: true

module StoreFront
  module Mixins
    module Trackable
      def self.included(base)
        base.extend(ClassMethods)
      end

      module ClassMethods
        def tracked_events
          @tracked_events ||= []
        end
      end

      def track(event)
        self.class.tracked_events << event
      end
    end
  end
end
