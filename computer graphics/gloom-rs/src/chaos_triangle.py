def chaos_triangle_matrix(outer_vertices, n):

    top_left_x = float((outer_vertices[0][0] + outer_vertices[2][0]) * 0.5)    
    top_right_x =  float((outer_vertices[0][0] + outer_vertices[1][0]) * 0.5) 
    bottom_center_x = float(outer_vertices[0][0])

    top_y = float((outer_vertices[0][1] + outer_vertices[1][1]) * 0.5) 
    bottom_y = float(outer_vertices[1][1])

    new_matrix = [\
        top_left_x, top_y, 0.0, \
        bottom_center_x, bottom_y, 0.0, \
        top_right_x, top_y, 0.0 \
    ]

    if n == 0: 
        return new_matrix

    #Top
    top_vertices =  [[outer_vertices[0][0], outer_vertices[0][1], 0.0], \
                 [top_right_x, top_y, 0.0], \
                 [top_left_x, top_y, 0.0]]
    new_matrix += chaos_triangle_matrix(top_vertices, n-1)
    #Right
    right_vertices =  [[top_right_x, top_y, 0.0], \
                 [outer_vertices[1][0],  outer_vertices[1][1], 0.0], \
                 [bottom_center_x, bottom_y, 0.0]]
    new_matrix += chaos_triangle_matrix(right_vertices, n-1)
    #Left
    left_vertices =  [[top_left_x, top_y, 0.0], \
                 [bottom_center_x, bottom_y, 0.0], \
                 [outer_vertices[2][0], outer_vertices[2][1], 0.0]]
    new_matrix += chaos_triangle_matrix(left_vertices, n-1)

    return new_matrix

outer_vertices =  [[0, 1], [1, -1], [-1, -1]]

chaos_array = chaos_triangle_matrix(outer_vertices, 6) #Second argument creates 3^n triangles

print("Coordinates:", chaos_array)
print("Indices:", list(range(int(len(chaos_array)/3))))
print("Number of triangles:", {len(chaos_array) / 9})
print("Number of indices:", {len(chaos_array) / 3})
