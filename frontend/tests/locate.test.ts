import { describe, expect, it } from "vitest";
import { locateSummary } from "../src/dashboard/locate";
import type { LocateResult, LocatedPhone } from "../src/shared/types";

const phone = (over: Partial<LocatedPhone> = {}): LocatedPhone =>
  ({ status: "ok", x_m: 1, y_m: 1, rms_m: 0.01, moved_m: null, moved: false, ...over });
const result = (over: Partial<LocateResult> = {}): LocateResult =>
  ({ success: true, applied: true, anchored: true, rms_m: 0.01, error: null,
     phones: { a: phone(), b: phone(), c: phone() }, ...over });

describe("locateSummary", () => {
  it("reports a clean placement and how to fix the orientation", () => {
    expect(locateSummary(result())).toBe("Placed 3 of 3 phones. If the map is turned or mirrored, use Turn or Mirror.");
  });

  it("names loose and failed phones by label", () => {
    const r = result({ phones: { a: phone(), b: phone({ status: "check" }), c: phone({ status: "failed" }) } });
    const text = locateSummary(r, (id) => id.toUpperCase());
    expect(text).toContain("Placed 2 of 3 phones.");
    expect(text).toContain("Check B: the fit is loose.");
    expect(text).toContain("Couldn't place C: drag it by hand.");
  });

  it("a check run says nothing moved, or which phone moved and by how much", () => {
    expect(locateSummary(result({ applied: false }))).toBe("No phone has moved.");
    const moved = result({ applied: false, phones: { a: phone(), b: phone({ moved: true, moved_m: 0.34 }), c: phone() } });
    expect(locateSummary(moved)).toBe("Moved: b (0.34 m). Put it back or locate again.");
  });

  it("warns when the layout could not be anchored to the laptop", () => {
    expect(locateSummary(result({ anchored: false }))).toContain("didn't hear the laptop");
  });

  it("passes the server's reason through on failure", () => {
    expect(locateSummary(result({ success: false, error: "Need at least 3 connected phones" })))
      .toBe("Need at least 3 connected phones");
  });
});
