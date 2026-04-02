"""AABB collision detection and resolution."""

import pygame
from entities.entity import Entity
from entities.movable_object import MovableObject


def resolve_entity_walls(entity: Entity, wall_rects: list[pygame.Rect]):
    """Push entity out of any overlapping walls.
    Resolves X and Y axes independently to allow wall-sliding."""
    half = entity.ENTITY_SIZE // 2

    # --- Resolve X axis ---
    # Move only X, keep Y at previous position
    prev_x = entity.x - entity.vx
    entity.rect.x = int(entity.x) - half
    entity.rect.y = int(entity.y - entity.vy) - half  # use old Y

    for wall in wall_rects:
        if entity.rect.colliderect(wall):
            if entity.vx > 0:
                entity.rect.right = wall.left
            elif entity.vx < 0:
                entity.rect.left = wall.right
            entity.x = float(entity.rect.x + half)

    # --- Resolve Y axis ---
    # Now apply Y with corrected X
    entity.rect.y = int(entity.y) - half

    for wall in wall_rects:
        if entity.rect.colliderect(wall):
            if entity.vy > 0:
                entity.rect.bottom = wall.top
            elif entity.vy < 0:
                entity.rect.top = wall.bottom
            entity.y = float(entity.rect.y + half)

    # Final sync
    entity.rect.x = int(entity.x) - half
    entity.rect.y = int(entity.y) - half


def resolve_entity_crates(entity: Entity, crates: list[MovableObject],
                          wall_rects: list[pygame.Rect]):
    """Handle entity pushing crates. If entity overlaps a crate, try to push it.
    If the crate can't move (blocked by wall/another crate), block the entity too."""
    for crate in crates:
        if not entity.rect.colliderect(crate.rect):
            continue

        # Determine push direction from overlap, not velocity
        overlap_left = entity.rect.right - crate.rect.left
        overlap_right = crate.rect.right - entity.rect.left
        overlap_top = entity.rect.bottom - crate.rect.top
        overlap_bottom = crate.rect.bottom - entity.rect.top

        min_x = min(overlap_left, overlap_right)
        min_y = min(overlap_top, overlap_bottom)

        # Push along the axis with least overlap
        push_dx = 0
        push_dy = 0
        if min_x < min_y:
            push_dx = 1 if overlap_left < overlap_right else -1
        else:
            push_dy = 1 if overlap_top < overlap_bottom else -1

        # Check if crate can move
        new_crate_rect = crate.push(push_dx, push_dy)
        blocked = False

        for wall in wall_rects:
            if new_crate_rect.colliderect(wall):
                blocked = True
                break

        if not blocked:
            for other_crate in crates:
                if other_crate is crate:
                    continue
                if new_crate_rect.colliderect(other_crate.rect):
                    blocked = True
                    break

        if not blocked:
            crate.apply_push(push_dx, push_dy)

        # Always push entity out of crate
        _push_entity_out_of_crate(entity, crate)


def _push_entity_out_of_crate(entity: Entity, crate: MovableObject):
    """Push entity out of crate overlap using minimum penetration."""
    if not entity.rect.colliderect(crate.rect):
        return

    overlap_left = entity.rect.right - crate.rect.left
    overlap_right = crate.rect.right - entity.rect.left
    overlap_top = entity.rect.bottom - crate.rect.top
    overlap_bottom = crate.rect.bottom - entity.rect.top

    min_overlap = min(overlap_left, overlap_right, overlap_top, overlap_bottom)

    half = entity.ENTITY_SIZE // 2
    if min_overlap == overlap_left:
        entity.rect.right = crate.rect.left
    elif min_overlap == overlap_right:
        entity.rect.left = crate.rect.right
    elif min_overlap == overlap_top:
        entity.rect.bottom = crate.rect.top
    else:
        entity.rect.top = crate.rect.bottom

    entity.x = float(entity.rect.x + half)
    entity.y = float(entity.rect.y + half)
