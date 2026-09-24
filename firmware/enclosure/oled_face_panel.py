"""
Rocky front face panel: 0.96" SSD1306 OLED holder + speaker grille.

Run:  python oled_face_panel.py     (needs `pip install cadquery`)
Out:  rocky_face_panel.step/.stl   full panel
      oled_fit_test.step/.stl      small coupon -- print this FIRST to check
                                   the module fits before printing the panel

Orientation: front face at Z=0 (print it face-down on the bed, no supports),
electronics side faces +Z. All dimensions in mm. Every module dimension below
is a typical-datasheet DEFAULT -- 0.96" breakout boards vary between sellers,
so measure yours with calipers and edit the MEASURE-ME block before printing
the full panel.
"""
import cadquery as cq

# ---------------------------------------------------------------- MEASURE-ME
PCB_W, PCB_H = 27.3, 27.8         # breakout board outline
HOLE_INSET = 2.0                  # PCB corner -> mounting-hole centre
GLASS_W, GLASS_H = 26.7, 19.3     # glass panel outline
GLASS_T = 1.9                     # glass + polarizer thickness above PCB face
GLASS_TOP_MARGIN = 3.5            # PCB top edge -> glass top edge
ACTIVE_W, ACTIVE_H = 21.74, 10.86  # lit pixel area (128x64)
ACTIVE_TOP_MARGIN = 2.0           # glass top edge -> top of lit area
PIN_EDGE = "bottom"               # PCB edge the 4-pin header is on: top/bottom
SPK_D = 28.0                      # speaker diameter (measure yours!)

# ------------------------------------------------------------ DESIGN CHOICES
PLATE_W, PLATE_H = 110.0, 80.0    # front panel outline
PLATE_R = 6.0                     # panel corner radius
FRONT_CHAMFER = 0.6               # soft edge on the visible face
SKIN = 1.2                        # material left in front of the glass
CLR = 0.4                         # total clearance added to pockets
PLATE_T = SKIN + GLASS_T + 0.1    # plate thickness (glass sits flush behind skin)
RIM_T, RIM_H = 1.6, 1.8           # wall that locates the PCB, and its height
WIN_MARGIN_X, WIN_MARGIN_Y = 1.0, 1.2  # window bigger than lit area (viewing angle)
PILOT_D, PILOT_DEPTH = 1.7, 2.0   # blind holes for M2x3 self-tapping screws
NOTCH_W = 14.0                    # cable exit in the rim, on the header edge
OLED_CENTER = (0.0, 18.0)         # PCB centre on the panel
SPK_CENTER = (0.0, -20.0)
SPK_RING_H = 4.0
GRILLE_HOLE, GRILLE_PITCH = 2.2, 3.5
CORNER_HOLE, CORNER_INSET = 2.7, 5.0  # panel-to-shell screws, countersunk

E = 0.01  # overlap so boolean cuts never leave zero-thickness skins


def box(cx, cy, w, h, z0, z1, fillet=0.0):
    s = cq.Workplane("XY").workplane(offset=z0).center(cx, cy).rect(w, h).extrude(z1 - z0)
    return s.edges("|Z").fillet(fillet) if fillet else s


def cyl(cx, cy, d, z0, z1):
    return cq.Workplane("XY").workplane(offset=z0).center(cx, cy).circle(d / 2).extrude(z1 - z0)


def add_oled_holder(plate, cx, cy):
    """Window, glass recess, PCB locating rim, cable notch, screw pilots."""
    glass_top = cy + PCB_H / 2 - GLASS_TOP_MARGIN
    glass_cy = glass_top - GLASS_H / 2
    active_cy = glass_top - ACTIVE_TOP_MARGIN - ACTIVE_H / 2

    # Glass recess from the rear, leaving SKIN in front; window through the skin.
    plate = plate.cut(box(cx, glass_cy, GLASS_W + CLR, GLASS_H + CLR, SKIN, PLATE_T + E))
    plate = plate.cut(box(cx, active_cy, ACTIVE_W + 2 * WIN_MARGIN_X,
                          ACTIVE_H + 2 * WIN_MARGIN_Y, -E, SKIN + E, fillet=1.0))

    # Rim around the PCB (PCB face rests on the plate's rear surface).
    outer = box(cx, cy, PCB_W + CLR + 2 * RIM_T, PCB_H + CLR + 2 * RIM_T, PLATE_T - E, PLATE_T + RIM_H)
    inner = box(cx, cy, PCB_W + CLR, PCB_H + CLR, PLATE_T - 2 * E, PLATE_T + RIM_H + E)
    plate = plate.union(outer.cut(inner))

    # Cable notch through the rim on the header edge.
    sign = -1 if PIN_EDGE == "bottom" else 1
    ny = cy + sign * (PCB_H + CLR + RIM_T) / 2
    plate = plate.cut(box(cx, ny, NOTCH_W, RIM_T + 1.0, PLATE_T, PLATE_T + RIM_H + E))

    # Screw pilots at the PCB mounting holes -- must clear the glass recess.
    for sx in (-1, 1):
        for sy in (-1, 1):
            hx = cx + sx * (PCB_W / 2 - HOLE_INSET)
            hy = cy + sy * (PCB_H / 2 - HOLE_INSET)
            in_glass = (abs(hx - cx) < (GLASS_W + CLR) / 2 + PILOT_D / 2 and
                        abs(hy - glass_cy) < (GLASS_H + CLR) / 2 + PILOT_D / 2)
            if in_glass:
                print(f"WARNING: mounting hole at ({hx:.1f},{hy:.1f}) overlaps the glass "
                      f"recess -- check GLASS_TOP_MARGIN / HOLE_INSET against your module.")
            plate = plate.cut(cyl(hx, hy, PILOT_D, PLATE_T - PILOT_DEPTH, PLATE_T + E))
    return plate


def add_speaker(plate, cx, cy):
    """Locating ring on the rear plus a hex-packed grille through the plate."""
    ring = cyl(cx, cy, SPK_D + CLR + 2 * RIM_T, PLATE_T - E, PLATE_T + SPK_RING_H)
    ring = ring.cut(cyl(cx, cy, SPK_D + CLR, PLATE_T - 2 * E, PLATE_T + SPK_RING_H + E))
    plate = plate.union(ring)

    limit = (SPK_D - 4.0) / 2 - GRILLE_HOLE / 2
    row_h = GRILLE_PITCH * 0.866
    n = int(limit / row_h) + 1
    for r in range(-n, n + 1):
        y = r * row_h
        x0 = (GRILLE_PITCH / 2) if r % 2 else 0.0
        for c in range(-n - 1, n + 2):
            x = x0 + c * GRILLE_PITCH
            if x * x + y * y <= limit * limit:
                plate = plate.cut(cyl(cx + x, cy + y, GRILLE_HOLE, -E, PLATE_T + E))
    return plate


def face_panel():
    plate = cq.Workplane("XY").rect(PLATE_W, PLATE_H).extrude(PLATE_T)
    plate = plate.edges("|Z").fillet(PLATE_R)
    plate = plate.faces("<Z").edges().chamfer(FRONT_CHAMFER)
    plate = add_oled_holder(plate, *OLED_CENTER)
    plate = add_speaker(plate, *SPK_CENTER)
    pts = [(sx * (PLATE_W / 2 - CORNER_INSET), sy * (PLATE_H / 2 - CORNER_INSET))
           for sx in (-1, 1) for sy in (-1, 1)]
    plate = plate.faces("<Z").workplane().pushPoints(pts).cskHole(CORNER_HOLE, CORNER_HOLE * 2, 90)
    return plate


def fit_test():
    w = PCB_W + CLR + 2 * RIM_T + 6
    h = PCB_H + CLR + 2 * RIM_T + 6
    plate = cq.Workplane("XY").rect(w, h).extrude(PLATE_T).edges("|Z").fillet(3.0)
    plate = plate.faces("<Z").edges().chamfer(FRONT_CHAMFER)
    return add_oled_holder(plate, 0.0, 0.0)


def check_and_export(solid, name):
    s = solid.val()
    bb = s.BoundingBox()
    n_solids = len(solid.solids().vals())
    print(f"{name}: valid={s.isValid()} solids={n_solids} volume={s.Volume():.0f} mm^3 "
          f"size={bb.xlen:.1f} x {bb.ylen:.1f} x {bb.zlen:.1f}")
    assert s.isValid() and n_solids == 1, f"{name} is not a single valid solid"
    cq.exporters.export(solid, f"{name}.step")
    cq.exporters.export(solid, f"{name}.stl", tolerance=0.01, angularTolerance=0.1)


if __name__ == "__main__":
    check_and_export(fit_test(), "oled_fit_test")
    check_and_export(face_panel(), "rocky_face_panel")
