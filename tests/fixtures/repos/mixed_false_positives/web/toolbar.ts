export type ToolbarAction = "save" | "cancel";

export class ToolbarState {
  private selectedAction: ToolbarAction | null = null;

  select(action: ToolbarAction): void {
    this.selectedAction = action;
  }

  current(): ToolbarAction | null {
    return this.selectedAction;
  }
}

export function renderToolbar(actions: ToolbarAction[]): string {
  return actions.map((action) => `<button>${action}</button>`).join("");
}
