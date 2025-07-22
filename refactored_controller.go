package controllers

// Refactored GetApplications method using the abstracted pagination service
func (c *ApiController) GetApplications() {
	params := c.GetQueryParams()
	
	// Extract pagination parameters from query params
	config := PaginationConfig{
		Owner:        params["owner"],
		Limit:        params["limit"],
		Page:         params["page"],
		Field:        params["field"],
		Value:        params["value"],
		SortField:    params["sortField"],
		SortOrder:    params["sortOrder"],
		Organization: params["organization"],
	}
	
	// Create pagination service and application service
	paginationService := NewPaginationService()
	applicationService := &ApplicationService{}
	
	// Handle the paginated query
	paginationService.HandlePaginatedQuery(c, config, applicationService)
}

// Example: UserService implementing ResourceService for User resources
type UserService struct{}

func (us *UserService) GetAll(owner string) (interface{}, error) {
	return object.GetUsers(owner)
}

func (us *UserService) GetAllByOrganization(owner, organization string) (interface{}, error) {
	return object.GetOrganizationUsers(owner, organization)
}

func (us *UserService) GetCount(owner, field, value string) (int64, error) {
	return object.GetUserCount(owner, field, value)
}

func (us *UserService) GetPaginated(owner string, offset, limit int, field, value, sortField, sortOrder string) (interface{}, error) {
	return object.GetPaginationUsers(owner, offset, limit, field, value, sortField, sortOrder)
}

func (us *UserService) GetMasked(resources interface{}, userId string) interface{} {
	if users, ok := resources.([]*object.User); ok {
		return object.GetMaskedUsers(users, userId)
	}
	return resources
}

// Example: GetUsers method using the abstracted pagination
func (c *ApiController) GetUsers() {
	params := c.GetQueryParams()
	
	config := PaginationConfig{
		Owner:        params["owner"],
		Limit:        params["limit"],
		Page:         params["page"],
		Field:        params["field"],
		Value:        params["value"],
		SortField:    params["sortField"],
		SortOrder:    params["sortOrder"],
		Organization: params["organization"],
	}
	
	paginationService := NewPaginationService()
	userService := &UserService{}
	
	paginationService.HandlePaginatedQuery(c, config, userService)
}

// Example: RoleService implementing ResourceService for Role resources
type RoleService struct{}

func (rs *RoleService) GetAll(owner string) (interface{}, error) {
	return object.GetRoles(owner)
}

func (rs *RoleService) GetAllByOrganization(owner, organization string) (interface{}, error) {
	return object.GetOrganizationRoles(owner, organization)
}

func (rs *RoleService) GetCount(owner, field, value string) (int64, error) {
	return object.GetRoleCount(owner, field, value)
}

func (rs *RoleService) GetPaginated(owner string, offset, limit int, field, value, sortField, sortOrder string) (interface{}, error) {
	return object.GetPaginationRoles(owner, offset, limit, field, value, sortField, sortOrder)
}

func (rs *RoleService) GetMasked(resources interface{}, userId string) interface{} {
	if roles, ok := resources.([]*object.Role); ok {
		return object.GetMaskedRoles(roles, userId)
	}
	return resources
}

// Example: GetRoles method using the abstracted pagination
func (c *ApiController) GetRoles() {
	params := c.GetQueryParams()
	
	config := PaginationConfig{
		Owner:        params["owner"],
		Limit:        params["limit"],
		Page:         params["page"],
		Field:        params["field"],
		Value:        params["value"],
		SortField:    params["sortField"],
		SortOrder:    params["sortOrder"],
		Organization: params["organization"],
	}
	
	paginationService := NewPaginationService()
	roleService := &RoleService{}
	
	paginationService.HandlePaginatedQuery(c, config, roleService)
}

// Helper method to extract common pagination config
func (c *ApiController) GetPaginationConfig() PaginationConfig {
	params := c.GetQueryParams()
	
	return PaginationConfig{
		Owner:        params["owner"],
		Limit:        params["limit"],
		Page:         params["page"],
		Field:        params["field"],
		Value:        params["value"],
		SortField:    params["sortField"],
		SortOrder:    params["sortOrder"],
		Organization: params["organization"],
	}
}

// Simplified controller methods using the helper
func (c *ApiController) GetApplicationsSimplified() {
	config := c.GetPaginationConfig()
	paginationService := NewPaginationService()
	applicationService := &ApplicationService{}
	paginationService.HandlePaginatedQuery(c, config, applicationService)
}

func (c *ApiController) GetUsersSimplified() {
	config := c.GetPaginationConfig()
	paginationService := NewPaginationService()
	userService := &UserService{}
	paginationService.HandlePaginatedQuery(c, config, userService)
}

func (c *ApiController) GetRolesSimplified() {
	config := c.GetPaginationConfig()
	paginationService := NewPaginationService()
	roleService := &RoleService{}
	paginationService.HandlePaginatedQuery(c, config, roleService)
}