package com.example.app.legacy {

  class LegacyWidget {
    def describe(): String = "legacy widget"
  }

  class LegacyGadget {
    def describe(): String = "legacy gadget"
  }

  object LegacyRegistry {
    def all(): List[String] = List(new LegacyWidget().describe(), new LegacyGadget().describe())
  }

}
