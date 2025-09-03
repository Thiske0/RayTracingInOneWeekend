use proc_macro2::TokenStream;
use quote::quote;
use syn::{
    parse_macro_input, parse_quote, punctuated::Punctuated, token::Comma, Fields, GenericParam,
    Generics, Ident, Item, ItemEnum, ItemStruct, Lifetime, LifetimeParam,
};

use crate::{
    common::{device_generics_ty_from_host, generate_fresh_lifetime, make_variants},
    derive_builder::builder_attributes::BuilderAttributes,
    derive_device_struct::derive_device_struct_inner,
    require_repr,
};

pub mod builder_attributes;

pub fn derive_builder(
    attrs: proc_macro::TokenStream,
    input: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let attrs = parse_macro_input!(attrs as BuilderAttributes);
    let input = parse_macro_input!(input as Item);
    let impl_block = derive_builder_inner(attrs, input);
    impl_block.unwrap_or_else(|e| e.to_compile_error()).into()
}

pub fn derive_builder_inner(attrs: BuilderAttributes, input: Item) -> syn::Result<TokenStream> {
    let builder_impl = match &input {
        Item::Struct(s) => derive_builder_struct(attrs, s),
        Item::Enum(e) => derive_builder_enum(attrs, e),
        _ => Err(syn::Error::new_spanned(
            input.clone(),
            "#[derive_builder] only works for structs or enums",
        )),
    }?;
    let struct_impl = derive_device_struct_inner(input)?;
    Ok(quote! {
        #struct_impl
        #builder_impl
    })
}

pub fn derive_builder_struct(
    attrs: BuilderAttributes,
    input: &ItemStruct,
) -> syn::Result<TokenStream> {
    require_repr(&input.attrs)?;
    let name = &input.ident;
    let device_name = Ident::new(&format!("{}Device", name), name.span());
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    let mut impl_generics = quote! { #impl_generics };
    let device_generics_ty = device_generics_ty_from_host(&input.generics);

    let new_lifetime = if let Some(lifetime) = attrs.lifetime {
        lifetime
    } else {
        let lifetime = generate_fresh_lifetime(&input.generics);
        impl_generics = add_lifetime(&lifetime, impl_generics);
        lifetime
    };

    let fields = match &input.fields {
        Fields::Named(fields) => Ok(&fields.named),
        _ => Err(syn::Error::new_spanned(
            input.fields.clone(),
            "#[derive_builder] for structs only works with named fields",
        )),
    }?;

    let no_copy_fields: Punctuated<_, Comma> = fields
        .iter()
        .filter(|field| {
            for attr in &field.attrs {
                if attr.path().is_ident("no_copy") {
                    return true;
                }
            }
            false
        })
        .collect();

    let fields: Punctuated<_, Comma> = fields
        .iter()
        .filter(|field| {
            for attr in &field.attrs {
                if attr.path().is_ident("host_only") || attr.path().is_ident("no_copy") {
                    return false;
                }
            }
            true
        })
        .collect();

    let fields_device = &fields
        .iter()
        .map(|field| {
            let field_name = field.ident.as_ref().unwrap();
            syn::Ident::new(&format!("{}_device", field_name), field_name.span())
        })
        .collect::<Vec<_>>();

    let struct_fields = fields
        .iter()
        .map(|field| {
            let field_name = &field.ident;
            quote! {
                #field_name: #field_name,
            }
        })
        .collect::<Vec<_>>();
    let struct_fields_no_copy = no_copy_fields
        .iter()
        .map(|field| {
            let field_name = &field.ident;
            let field_ty = &field.ty;
            quote! {
                #field_name: <#field_ty>::default(),
            }
        })
        .collect::<Vec<_>>();

    let struct_fields_device = fields
        .iter()
        .zip(fields_device)
        .map(|(field, field_device)| {
            let field_name = &field.ident;
            quote! {
                #field_name: #field_device,
            }
        })
        .collect::<Vec<_>>();

    let build_device_fields = fields.iter().zip(fields_device).map(|(field, field_device)| {
                let field_name = &field.ident;
                quote! {
                    let (#field_device, #field_name, buffers) = unsafe { self.#field_name.build_device_inner(stream, cache)? }.split();
                    device_buffer_list.combine(buffers);
                }
            });

    let copy_back_fields = fields.iter().map(|field| {
                let field_name = &field.ident;
                quote! {
                    let field_ptr = base_ptr + std::mem::offset_of!(#device_name #device_generics_ty, #field_name) as u64;
                    let field_device = unsafe { cust::memory::DeviceBox::from_raw(field_ptr) };
                    self.#field_name.copy_back(&field_device)?;
                }
            });

    let impl_block = quote! {
        #[cfg(not(target_os = "cuda"))]
        impl #impl_generics gpu_builder::Builder<#new_lifetime> for #name #ty_generics #where_clause {
            type Output = #device_name #device_generics_ty;
            unsafe fn build_device_inner(
                self,
                stream: &#new_lifetime cust::stream::Stream,
                cache: &mut gpu_builder::Cache<#new_lifetime>,
            ) -> cust::error::CudaResult<gpu_builder::BuildResult<#new_lifetime, Self>> {
                let mut device_buffer_list = gpu_builder::DeviceBufferList::new();
                #(#build_device_fields)*
                let result_device = #device_name {
                    #(#struct_fields_device)*
                };
                let result_host = #name {
                    #(#struct_fields)*
                    #(#struct_fields_no_copy)*
                };
                Ok(gpu_builder::BuildResult::new(result_device, result_host, stream, device_buffer_list))
            }

            fn copy_back(
                &mut self,
                c_device: &cust::memory::DeviceBox<<Self as gpu_builder::Builder<#new_lifetime>>::Output>,
            ) -> Result<(), cust::error::CudaError> {
                let base_ptr = c_device.as_device_ptr().as_raw();

                #(#copy_back_fields)*

                Ok(())
            }
        }
    };
    Ok(impl_block)
}

pub fn derive_builder_enum(attrs: BuilderAttributes, input: &ItemEnum) -> syn::Result<TokenStream> {
    require_repr(&input.attrs)?;

    let name = &input.ident;
    let device_name = Ident::new(&format!("{}Device", name), name.span());
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    let mut impl_generics = quote! { #impl_generics };
    let device_generics_ty = device_generics_ty_from_host(&input.generics);

    let new_lifetime = if let Some(lifetime) = attrs.lifetime {
        lifetime
    } else {
        let lifetime = generate_fresh_lifetime(&input.generics);
        impl_generics = add_lifetime(&lifetime, impl_generics);
        lifetime
    };

    let variants_build_device = make_variants(&input.variants, |field_names, make_enum| {
        let field_device_names = &field_names
            .iter()
            .map(|field_name| syn::Ident::new(&format!("{}_device", field_name), field_name.span()))
            .collect::<Vec<_>>();
        let build_device = field_names
            .iter()
            .zip(field_device_names)
            .map(|(field_name, field_device_name)| {
                quote! {
                    let (#field_device_name, #field_name, buffers) = #field_name.build_device_inner(stream, cache)?.split();
                    device_buffer_list.combine(buffers);
                }
            })
            .collect::<Vec<_>>();
        let field_device_name_tokens = field_device_names
            .iter()
            .map(|f| quote! { #f })
            .collect::<Vec<_>>();
        let field_name_tokens = field_names
            .iter()
            .map(|f| quote! { #f })
            .collect::<Vec<_>>();
        let device_enum_variant = make_enum(&device_name, field_device_name_tokens);
        let host_enum_variant = make_enum(&name, field_name_tokens);
        quote! {
            #(#build_device)*
            (#device_enum_variant, #host_enum_variant)
        }
    });

    let variants_copy_back = input.variants.iter().map(|variant| {
        let variant_name = &variant.ident;
        match &variant.fields {
            Fields::Named(fields) => {
                let field_names: Vec<_> = fields.named.iter().map(|f| &f.ident).collect();
                let field_types: Vec<_> = fields.named.iter().map(|f| &f.ty).collect();
                let fields_build = field_names.iter().map(|name| {
                    quote! {
                        let field_ptr = base_ptr + std::mem::offset_of!(#device_name #device_generics_ty, #variant_name.#name) as u64;
                        let field_device = unsafe { cust::memory::DeviceBox::from_raw(field_ptr) };
                        #name.copy_back(&field_device)?;
                    }
                });
                let fields_convert = field_names.iter().zip(field_types).map(|(name, ty)| {
                    quote! {
                        unsafe { std::ptr::read(Into::<&#ty>::into(&#name) as *const #ty) }
                    }
                });
                quote! {
                    #device_name::#variant_name { #(#field_names),* } => {
                        *self = #name::#variant_name{#(#field_names: #fields_convert),*};
                        match *self {
                            #name::#variant_name{#(ref mut #field_names),*} => {
                                #(#fields_build)*
                            }
                            _ => unreachable!(),
                        };
                    },
                }
            }
            Fields::Unnamed(fields) => {
                let field_names: Vec<_> = fields
                    .unnamed
                    .iter()
                    .enumerate()
                    .map(|(i, _f)| Ident::new(&format!("field_{}", i), name.span()))
                    .collect();
                let field_types: Vec<_> = fields.unnamed.iter().map(|f| &f.ty).collect();
                let indexes: Vec<_> =
                    fields.unnamed.iter().enumerate().map(|(i, _f)| syn::Index::from(i)).collect();
                let fields_build = field_names.iter().zip(indexes).map(|(name, index)| {
                    quote! {
                        let field_ptr = base_ptr + std::mem::offset_of!(#device_name #device_generics_ty, #variant_name.#index) as u64;
                        let field_device = unsafe { cust::memory::DeviceBox::from_raw(field_ptr) };
                        #name.copy_back(&field_device)?;
                    }
                });
                let fields_convert = field_names.iter().zip(field_types).map(|(name, ty)| {
                    quote! {
                        unsafe { std::ptr::read(Into::<&#ty>::into(&#name) as *const #ty) }
                    }
                });
                quote! {
                    #device_name::#variant_name(#(#field_names),*) => {
                        *self = #name::#variant_name(#(#fields_convert),*);
                        match *self {
                            #name::#variant_name(#(ref mut #field_names),*) => {
                                #(#fields_build)*
                            }
                            _ => unreachable!(),
                        };
                    },
                }
            }
            Fields::Unit => {
                quote! {
                    #device_name::#variant_name => {
                        *self = #name::#variant_name;
                    },
                }
            }
        }
    });

    let impl_block = quote! {
        #[cfg(not(target_os = "cuda"))]
        impl #impl_generics gpu_builder::Builder<#new_lifetime> for #name #ty_generics #where_clause {
            type Output = #device_name #device_generics_ty;
            unsafe fn build_device_inner(
                self,
                stream: &#new_lifetime cust::stream::Stream,
                cache: &mut gpu_builder::Cache<#new_lifetime>,
            ) -> cust::error::CudaResult<gpu_builder::BuildResult<#new_lifetime, Self>> {
                let mut device_buffer_list = gpu_builder::DeviceBufferList::new();
                let (result_device, result_host) = match self {
                    #(#variants_build_device)*
                };
                Ok(gpu_builder::BuildResult::new(result_device, result_host, stream, device_buffer_list))
            }

            fn copy_back(
                &mut self,
                device_box: &cust::memory::DeviceBox<<Self as gpu_builder::Builder<#new_lifetime>>::Output>,
            ) -> cust::error::CudaResult<()> {
                let host_value = device_box.as_host_value()?;
                let base_ptr = device_box.as_device_ptr().as_raw();
                match host_value {
                    #(#variants_copy_back)*
                }
                Ok(())
            }
        }
    };
    Ok(impl_block)
}

fn add_lifetime(lifetime: &Lifetime, impl_generics: TokenStream) -> TokenStream {
    let mut generics: Generics = parse_quote! { #impl_generics };
    generics.params.insert(
        0,
        GenericParam::Lifetime(LifetimeParam::new(lifetime.clone())),
    );
    let impl_generics = generics.split_for_impl().0;
    quote! { #impl_generics }
}
