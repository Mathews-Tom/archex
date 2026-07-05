function openTicketForm(category) {
  return { category, status: "draft", createdAt: Date.now() };
}

function escalateTicket(ticket, reason) {
  return { ...ticket, status: "escalated", escalationReason: reason };
}

function routeTicketToQueue(ticket) {
  const queueByCategory = { billing: "finance-team", technical: "on-call-engineer" };
  return queueByCategory[ticket.category] ?? "general-queue";
}

function sendSatisfactionSurvey(ticket) {
  return { ticketId: ticket.id, surveySentAt: Date.now() };
}

function archiveTicketTranscript(ticket, transcript) {
  return { ticketId: ticket.id, transcript, archivedAt: Date.now() };
}
