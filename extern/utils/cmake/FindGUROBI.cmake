# FindGUROBI.cmake -- this module finds the GUROBI optimizer
#
# Users might need to set/overwrite GUROBIDIR

# Find GUROBI headers and library
find_path(GUROBI_INCLUDE_DIR "gurobi_c++.h" PATHS ${GUROBIDIR}/linux64/include/ ${GUROBIDIR}/armlinux64/include/)
find_library(GUROBI_LIBRARY NAMES "libgurobi_c++.a" PATHS ${GUROBIDIR}/linux64/lib/ ${GUROBIDIR}/armlinux64/lib/)
find_library(GUROBI_SHARED_LIBRARY NAMES "libgurobi.so" PATHS ${GUROBIDIR}/linux64/lib/ ${GUROBIDIR}/armlinux64/lib/)

mark_as_advanced(GUROBI_INCLUDE_DIR GUROBI_LIBRARY GUROBI_SHARED_LIBRARY)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GUROBI DEFAULT_MSG GUROBI_LIBRARY GUROBI_INCLUDE_DIR GUROBI_SHARED_LIBRARY)

if (GUROBI_FOUND)
	# Create imported target Gurobi::Gurobi
	add_library(Gurobi::Gurobi SHARED IMPORTED)
	set_target_properties(Gurobi::Gurobi PROPERTIES
		IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
		IMPORTED_LOCATION "${GUROBI_LIBRARY}"
		INTERFACE_INCLUDE_DIRECTORIES "${GUROBI_INCLUDE_DIR}"
		INTERFACE_LINK_LIBRARIES "${GUROBI_SHARED_LIBRARY}"
	)
endif()
