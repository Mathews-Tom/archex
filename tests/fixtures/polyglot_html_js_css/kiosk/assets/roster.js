function renderTerminalList(entries) {
  return entries.map((entry) => `<li>${entry.name}</li>`).join("");
}

function lookupStaffEntry(roster, badgeNumber) {
  return roster.find((entry) => entry.badgeNumber === badgeNumber) ?? null;
}

function refreshKioskDirectory(terminalId, directorySnapshot) {
  return {
    terminalId,
    entries: directorySnapshot.entries,
    syncedAt: directorySnapshot.syncedAt,
  };
}

function requestReceptionCallback(entry, floorNumber) {
  return {
    badgeNumber: entry.badgeNumber,
    floorNumber,
    requestedAt: Date.now(),
  };
}

function filterByDepartment(entries, departmentName) {
  return entries.filter((entry) => entry.department === departmentName);
}
