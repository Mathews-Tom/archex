/* Generic third-party page-timing/telemetry beacon, vendored as-is. Not
   referenced by any page in this fixture — present only to give the corpus
   enough bulk that a benchmark token budget cannot trivially fit the whole
   repo. Unrelated in vocabulary to every other file here on purpose. */

(function (window) {
  "use strict";

  var QUEUE = [];
  var FLUSH_INTERVAL_MS = 5000;
  var ENDPOINT = "/collect";

  function generateSessionToken() {
    var alphabet = "abcdefghijklmnopqrstuvwxyz0123456789";
    var token = "";
    for (var i = 0; i < 24; i += 1) {
      token += alphabet[Math.floor(Math.random() * alphabet.length)];
    }
    return token;
  }

  function recordNavigationTiming() {
    if (!window.performance || !window.performance.timing) {
      return null;
    }
    var timing = window.performance.timing;
    return {
      dnsLookupMs: timing.domainLookupEnd - timing.domainLookupStart,
      tcpConnectMs: timing.connectEnd - timing.connectStart,
      responseMs: timing.responseEnd - timing.requestStart,
      domInteractiveMs: timing.domInteractive - timing.navigationStart,
    };
  }

  function enqueueBeacon(eventName, payload) {
    QUEUE.push({
      event: eventName,
      payload: payload || {},
      capturedAt: Date.now(),
    });
  }

  function flushQueue() {
    if (QUEUE.length === 0) {
      return;
    }
    var batch = QUEUE.splice(0, QUEUE.length);
    if (navigator.sendBeacon) {
      navigator.sendBeacon(ENDPOINT, JSON.stringify(batch));
      return;
    }
    var request = new XMLHttpRequest();
    request.open("POST", ENDPOINT, true);
    request.setRequestHeader("Content-Type", "application/json");
    request.send(JSON.stringify(batch));
  }

  function scheduleFlush() {
    window.setInterval(flushQueue, FLUSH_INTERVAL_MS);
    window.addEventListener("beforeunload", flushQueue);
  }

  function detectViewportBucket() {
    var width = window.innerWidth || document.documentElement.clientWidth;
    if (width < 480) {
      return "handset";
    }
    if (width < 1024) {
      return "tablet";
    }
    return "desktop";
  }

  function initTelemetry() {
    var sessionToken = generateSessionToken();
    enqueueBeacon("session_start", {
      sessionToken: sessionToken,
      viewportBucket: detectViewportBucket(),
      timing: recordNavigationTiming(),
    });
    scheduleFlush();
  }

  window.__vendorTelemetry = {
    init: initTelemetry,
    enqueueBeacon: enqueueBeacon,
    flushQueue: flushQueue,
  };
})(window);
