
def calculate_area(x,y):
    return 0.5*np.abs((x[:-1]*y[1:]-x[1:]*y[:-1]).sum()+x[-1]*y[0]-x[0]*y[-1])

# from OpticalElement
    # does not take curvature into account!
    def determine_quad_area_old(self,quad_z_indices,quad_r_indices):
        points = self.retrieve_single_quad_edge_points(quad_z_indices,quad_r_indices,return_ind_array=True)
        return calculate_area(self.z[points],self.r[points])

    def plot_mesh_segments(self,segments,quads_on=True):
        '''
        Plots mesh (coarse or fine) from a list of segments (Point1,Point2).

        Parameters:
            segments : list
                list of all segments to plot
            quads_on : boolean
                optional flag to also plot quads
        '''
        for segment in segments:
            plt.plot([segment[0].z,segment[1].z],[segment[0].r,segment[1].r],color='m')
        self.add_quads_to_plot() if quads_on else 0
        plt.xlabel("z (mm)")
        plt.ylabel("r (mm)")
        plt.gca().set_aspect('equal')
        plt.show()

