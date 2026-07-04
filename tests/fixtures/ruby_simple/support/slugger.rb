# frozen_string_literal: true

module StoreFront
  module Slugger
    SEPARATOR = "-"

    def self.slugify(value)
      value.to_s.downcase.gsub(/[^a-z0-9]+/, SEPARATOR).gsub(/^-|-$/, "")
    end
  end
end
